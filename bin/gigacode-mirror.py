#!/usr/bin/env python3
"""gigacode-mirror.py — spejl `code__gigacode__main` fra hus-Qdrant :16343 til læse-spejlet :16333.

KANARIE=GIGACODE-INGEST-20261105

HVORFOR (KOBLINGSPLAN-16343-TIL-16333.md §2): helix-instans 18207 (container hippocampus-helix) læser
`HIPPO_QDRANT_URL=http://localhost:16333` = container hippo-qdrant, som er en ANDEN Qdrant end hus-Qdrant
:16343 (hippo-hus-hippo-qdrant-1). Gigakode skriver kun til :16343 og afviser alt andet
(`assert_hus_qdrant`, gigakode.py:326-332) — den regel omgås IKKE her: dette script læser :16343 og
skriver :16333, og det rører aldrig gigakodes skrivevej.

HVAD DER SKAL VÆRE SANDT FOR AT 18207 KAN LÆSE (kilde, læst i samme arbejdsgang):
  · rag-collections.js:99-105 + :237-242  `code__` er public (PUBLIC_COLLECTION_PREFIXES) → ikke denied.
  · qdrant.js:798-801                     generation-gate gælder kun `code__*`.
  · qdrant.js:851-857 + isGenerationVisible:912-925
        post i `helix_branch_manifest` med `collection_name`-match; hvert point skal bære
        `generation` = `active_generation[point.source_path]` og `branch` = manifestets branch.
  · qdrant.js:887-890                     filteret matcher på `source_path` + `generation` + `branch`.
  · helix-20261031 har INGEN ingest der skriver `source_path` på code__-points; gigakodes payload bruger
        `path`. Derfor skriver dette script BEGGE felter: `path` (uændret, gigakodes) og `source_path`
        (= `path`), så både den målte gate-kode og en fremtidig `path`-baseret gate matcher.

GENERATION-VALG (dokumenteret, fordi opgaven kræver et valg):
  Gigakode har ingen per-fil version i sit point-payload (kun `sha256` pr. chunk, som ikke er filens
  sha256). Filens sha256 findes i gigakodes state: `state/<collection>/<repo>.json` → `files[rel]["h"]`.
  · Er state tilgængelig: `generation = files[rel]["h"][:16]` pr. fil — PRÆCIS den konvention
    qdrant.js:780 dokumenterer ("per point: filens sha256[:16]"), og `active_generation` bliver
    {path: sha16} pr. fil. Det er den stærkeste variant: en delvis ny ingest kan ikke gøre gamle
    punkter synlige.
  · Er state IKKE tilgængelig: ÉN samlet version pr. collection, `sha8(repo|branch|commit_sha)`
    (8 hex = 32 bit, deterministisk), og `active_generation = {path: version}` for ALLE unikke `path`.
    Dokumenteret svaghed: ved en delvis ingest (mid_ingest=true) kan gamle punkter for enkelte filer
    blive synlige, indtil næste komplette kørsel. Manifestet bærer `generation_mode` så det kan ses.
  Valget logges i hver kørsel som `generation_mode=state|collection`.

IDEMPOTENS:
  · commit_sha (fra kildens manifest-post, ellers kildens state, ellers `stable_tree_sha` af punkternes
    (path, sha256)-par) sammenlignes med `state/<collection>/<repo>.json` i mirror-state-dir.
  · Uændret → `unchanged`, 0 skriv (hverken points eller manifest).
  · Ændret → sammenlign pr. fil: kildens `sha256`-mængde pr. `path` mod spejlets. Ændrede/slettede
    filer: delete (filter repo+path) + upsert kun de ændrede punkter. Uændrede filer røres ikke.
  · Spejlets egen tilstand (commit_sha + pr.-fil sha256-fingeraftryk) skrives sidst, atomisk.

FAIL-CLOSED GUARDS (før nogen request):
  · --source skal være :16343 (og 127.0.0.1/localhost). Alt andet afvises.
  · --target skal være :16333. Alt andet afvises — inkl. :6333, :6340, :8700 og helios-værter.
  · collection skal starte med `code__`: kun `code__*` er public OG generation-gated i helix
    (rag-collections.js:99-105, qdrant.js:798-801). Uden præfikset ville en læser enten deny'e den
    (default-deny) eller læse den uden gate — begge er forkerte for et spejl af kode-RAG.

INGEN hemmeligheder: ingen tokens, ingen API-nøgler; Qdrant-URL'er kommer fra CLI/env.

KANARIE I HVER LOGLINJE: alle loglinjer bærer `KANARIE=GIGACODE-INGEST-20261105`.

FJERNELSE: filen er selvstændig og skriver kun til :16333 (collection + manifest-post) og til
--state-dir. Slet scriptet + state-filen; spejlet fjernes med
`DELETE :16333/collections/code__gigacode__main` og manifest-punktet.

Tests: tests/test_gigacode_mirror.py (offline, mock-HTTP, ingen netkald).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

KANARIE = "GIGACODE-INGEST-20261105"
VECTOR_SIZE = 2560                     # FROZEN (embed-qwen3) — samme som gigakode.py:56
DISTANCE = "Cosine"
SCROLL_BATCH = 256
UPSERT_BATCH = 128
MANIFEST_COLLECTION = "helix_branch_manifest"
# == lib/rag-collections.js BRANCH_MANIFEST_NS (rag-collections.js:80) og b1_manifest_bridge.py:26
BRANCH_MANIFEST_NS = uuid.UUID("2f8d4c61-7a3e-5b90-8d14-6c2fa9e07b53")
SOURCE_PORT = 16343
TARGET_PORT = 16333
ALLOWED_HOSTS = ("127.0.0.1", "localhost")
FORBIDDEN_PORTS = (6333, 6340, 8700)   # Helios/system-Qdrant — må aldrig berøres (gigakode.py:322)
LOG_PATH = os.environ.get("GIGAKODE_MIRROR_LOG", "/tmp/gigakode-mirror.log")


# --------------------------------------------------------------------------- log
def ts_iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else time.time()))


def log_line(**kv) -> None:
    """Hver loglinje bærer KANARIE. Skrives til stdout OG til GIGAKODE_MIRROR_LOG (append, aldrig fatal)."""
    parts = [ts_iso(), f"KANARIE={KANARIE}"] + [f"{k}={str(v).replace(chr(10), ' ')}" for k, v in kv.items()]
    line = " ".join(parts)
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # logfilen må ikke vælte spejlingen; stdout-linjen er stadig beviset


# --------------------------------------------------------------------------- guards
class GuardError(RuntimeError):
    """Fail-closed: afvist før nogen request er sendt."""


def _parse(url: str) -> urllib.parse.ParseResult:
    try:
        return urllib.parse.urlparse(url)
    except ValueError as exc:
        raise GuardError(f"ugyldig URL {url!r}: {exc}") from exc


def assert_source(url: str) -> str:
    """Kilden SKAL være hus-Qdrant :16343 (gigakode.py:326-332). Alt andet afvises."""
    u = _parse(url)
    host = (u.hostname or "").lower()
    if "helios" in host:
        raise GuardError(f"source={url!r} afvist: helios-vært (Helios-forbud)")
    if host not in ALLOWED_HOSTS:
        raise GuardError(f"source={url!r} afvist: værten skal være 127.0.0.1/localhost")
    if u.port in FORBIDDEN_PORTS:
        raise GuardError(f"source={url!r} afvist: port {u.port} er forbudt (Helios/system-Qdrant)")
    if u.port != SOURCE_PORT:
        raise GuardError(f"source={url!r} afvist: kilden skal være hus-Qdrant port {SOURCE_PORT}")
    if u.scheme not in ("http", "https"):
        raise GuardError(f"source={url!r} afvist: skemaet skal være http/https")
    return f"{u.scheme}://{host}:{u.port}".rstrip("/")


def assert_target(url: str) -> str:
    """Spejlet SKAL være :16333 (container hippo-qdrant). :16343 som target afvises (skrive-sandhed)."""
    u = _parse(url)
    host = (u.hostname or "").lower()
    if "helios" in host:
        raise GuardError(f"target={url!r} afvist: helios-vært (Helios-forbud)")
    if host not in ALLOWED_HOSTS:
        raise GuardError(f"target={url!r} afvist: værten skal være 127.0.0.1/localhost")
    if u.port in FORBIDDEN_PORTS:
        raise GuardError(f"target={url!r} afvist: port {u.port} er forbudt (Helios/system-Qdrant)")
    if u.port != TARGET_PORT:
        raise GuardError(f"target={url!r} afvist: spejlet skal være port {TARGET_PORT}")
    if u.scheme not in ("http", "https"):
        raise GuardError(f"target={url!r} afvist: skemaet skal være http/https")
    return f"{u.scheme}://{host}:{u.port}".rstrip("/")


def assert_code_collection(name: str) -> str:
    """Kun `code__*` er public OG generation-gated i helix (rag-collections.js:99-105, qdrant.js:798-801)."""
    n = (name or "").strip()
    if not n.startswith("code__"):
        raise GuardError(
            f"collection={n!r} afvist: kræver præfiks 'code__' — kun code__* er public i helix' "
            "allowlist OG eneste præfiks der er generation-gated (qdrant.js:798-801). "
            "Uden præfikset ville en læser enten deny'e den (default-deny) eller læse uden gate.")
    if "/" in n or " " in n:
        raise GuardError(f"collection={n!r} afvist: ugyldigt tegn")
    return n


# --------------------------------------------------------------------------- transport (injicerbar)
class Http:
    """Tynd urllib-wrapper. Kan erstattes i tests (se tests/test_gigacode_mirror.py)."""

    def __init__(self, base: str, timeout: int = 120):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.calls: list[tuple[str, str]] = []   # (method, path) — tælles af --dry-run-testen

    def req(self, method: str, path: str, body=None, timeout: int | None = None) -> dict:
        self.calls.append((method, path))
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"{self.base}{path}", data=data,
                                     headers={"Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}


# --------------------------------------------------------------------------- helpers (rene funktioner)
def stable_tree_sha(points: list[dict]) -> str:
    """Deterministisk commit_sha når kilden ikke har nogen: sha256 over sorterede (path, sha256)-par."""
    pairs = sorted({(str(p.get("path") or ""), str(p.get("sha256") or "")) for p in points})
    blob = "\n".join(f"{path}\x00{sha}" for path, sha in pairs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def collection_version(repo: str, branch: str, commit_sha: str) -> str:
    """ÉN samlet generation pr. collection: sha8(repo|branch|commit_sha) — deterministisk."""
    blob = f"{repo}|{branch}|{commit_sha}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def file_generation(file_sha256: str | None) -> str | None:
    """Per-fil version = filens sha256[:16] (samme konvention som qdrant.js:780)."""
    s = str(file_sha256 or "").strip()
    return s[:16] if len(s) >= 16 else None


def manifest_point_id(collection: str) -> str:
    """uuid5(BRANCH_MANIFEST_NS, '<repo>:<branch>') som github_branch_ingest.py — repo/branch udledt af
    collection-navnet `code__<repo>__<branch>` (repo må ikke indeholde '__' i gigakodes navnekonvention)."""
    inner = collection[len("code__"):]
    if "__" not in inner:
        raise GuardError(f"collection={collection!r} afvist: kan ikke udlede repo__branch")
    repo, branch = inner.rsplit("__", 1)
    if not repo or not branch:
        raise GuardError(f"collection={collection!r} afvist: tomt repo/branch")
    return str(uuid.uuid5(BRANCH_MANIFEST_NS, f"{repo}:{branch}"))


def file_fingerprint(points: list[dict]) -> dict[str, str]:
    """{path: sha256-mængde-hash} — hvad spejlet faktisk indeholder pr. fil (til idempotens)."""
    acc: dict[str, list[str]] = {}
    for p in points:
        acc.setdefault(str(p.get("path") or ""), []).append(str(p.get("sha256") or ""))
    return {path: hashlib.sha256("\n".join(sorted(shas)).encode()).hexdigest() for path, shas in acc.items()}


def source_file_fingerprint(points: list[dict]) -> dict[str, str]:
    """Samme form som file_fingerprint, beregnet på KILDENS punkter (sammenligningsgrundlag)."""
    return file_fingerprint(points)


def build_mirror_point(point: dict, branch: str, generation: str) -> dict:
    """Kopiér punktet 1:1 og tilføj de to gate-felter. `path` bevares; `source_path` = `path`."""
    payload = dict(point.get("payload") or {})
    path = str(payload.get("path") or "")
    payload["branch"] = branch
    payload["generation"] = generation
    payload["source_path"] = path
    return {"id": point.get("id"), "vector": point.get("vector"), "payload": payload}


def collection_create_body(source_config: dict) -> dict:
    """Collection-config kopieret fra kilden (dim 2560, Cosine, on_disk) — verificeret mod kilden."""
    params = ((source_config or {}).get("config") or {}).get("params") or {}
    vectors = params.get("vectors") or {}
    size = vectors.get("size") if isinstance(vectors, dict) else None
    distance = vectors.get("distance") if isinstance(vectors, dict) else None
    if size != VECTOR_SIZE:
        raise GuardError(f"kildens collection har vector size {size!r}, kræver {VECTOR_SIZE} (frozen)")
    if distance != DISTANCE:
        raise GuardError(f"kildens collection har distance {distance!r}, kræver {DISTANCE}")
    body = {
        "vectors": {"size": VECTOR_SIZE, "distance": DISTANCE, "on_disk": True},
        "hnsw_config": {"on_disk": True},
        "on_disk_payload": True,
    }
    hnsw = ((source_config or {}).get("config") or {}).get("hnsw_config") or {}
    if hnsw.get("m"):
        body["hnsw_config"]["m"] = int(hnsw["m"])
    if hnsw.get("ef_construct"):
        body["hnsw_config"]["ef_construct"] = int(hnsw["ef_construct"])
    return body


def build_manifest_payload(*, repo: str, source_repo: str, branch: str, collection: str, commit_sha: str,
                           points_count: int, files_count: int, active_generation: dict[str, str],
                           generation_mode: str, complete: bool, errors: int, mid_ingest: bool,
                           canary: str) -> dict:
    """Manifest-post med alle felter målt fra :16343 (46 poster, 2026-11-05) + generation-felterne
    (active_generation, generation_mode) som helix' gate læser (qdrant.js:805-830)."""
    return {
        "repo": repo,
        "source_repo": source_repo,
        "branch": branch,
        "collection_name": collection,
        "commit_sha": commit_sha,
        "points_count": int(points_count),
        "files_count": int(files_count),
        "updated_at": ts_iso() + "+00:00",   # lokal tid i ISO-form; samme format som de målte poster
        "legacy": False,
        "complete": bool(complete),
        "errors": int(errors),
        "mid_ingest": bool(mid_ingest),
        "canary": canary,
        "active_generation": dict(active_generation),
        "generation_mode": generation_mode,
    }


REQUIRED_MANIFEST_FIELDS = (
    "repo", "source_repo", "branch", "collection_name", "commit_sha", "points_count", "files_count",
    "updated_at", "legacy", "complete", "errors", "mid_ingest", "canary", "b1_namespace",
    "active_generation",
)


# --------------------------------------------------------------------------- state
def safe_name(repo: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", repo)


def state_path(state_dir: str, collection: str, repo: str) -> str:
    return os.path.join(state_dir, collection, f"{safe_name(repo)}.json")


def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.loads(fh.read())
    except (OSError, ValueError):
        return {}


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def gigakode_state(state_dir: str, collection: str, repo: str) -> dict:
    """Gigakodes egen state (`<state-dir>/<collection>/<repo>.json`).

    state_dir er gigakodes ROD (gigakode.py:438-440 gør selv `root/<collection>`); derfor står
    collection ikke igen i stien her. Læses read-only; findes den ikke → {}."""
    return load_json(os.path.join(state_dir, f"{safe_name(repo)}.json"))


# --------------------------------------------------------------------------- mirror
def read_source_manifest(http: Http, collection: str) -> dict | None:
    """Kildens manifest-post (samme filter som qdrant.js:809-812). Ingen post → None."""
    j = http.req("POST", f"/collections/{MANIFEST_COLLECTION}/points/scroll", {
        "limit": 5, "with_payload": True, "with_vector": False,
        "filter": {"must": [{"key": "collection_name", "match": {"value": collection}}]},
    })
    pts = ((j or {}).get("result") or {}).get("points") or []
    return (pts[0].get("payload") or {}) if pts else None


def scroll_all(http: Http, collection: str) -> list[dict]:
    """Læs ALLE punkter (payload + vector) i batches. Bevis: tæller kildens points/count og sammenligner."""
    out: list[dict] = []
    offset = None
    while True:
        body = {"limit": SCROLL_BATCH, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        j = http.req("POST", f"/collections/{collection}/points/scroll", body)
        res = (j or {}).get("result") or {}
        pts = res.get("points") or []
        out.extend(pts)
        offset = res.get("next_page_offset")
        if not pts or offset is None:
            break
    return out


def source_point_count(http: Http, collection: str) -> int:
    j = http.req("POST", f"/collections/{collection}/points/count", {"exact": True})
    return int(((j or {}).get("result") or {}).get("count") or 0)


def ensure_target_collection(http: Http, collection: str, source_config: dict, target_label: str = "") -> bool:
    """Opret spejlets collection hvis den mangler (config kopieret fra kilden). Returnerer created."""
    try:
        info = http.req("GET", f"/collections/{collection}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        info = None
    if info:
        vec = ((((info.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors")) or {}
        size = vec.get("size") if isinstance(vec, dict) else None
        if size and size != VECTOR_SIZE:
            raise GuardError(f"spejlets {collection} har vector size {size}, kræver {VECTOR_SIZE} (frozen)")
        return False
    http.req("PUT", f"/collections/{collection}", collection_create_body(source_config))
    log_line(event="collection_created", collection=collection, target=target_label, dim=VECTOR_SIZE,
             distance=DISTANCE)
    return True


def delete_paths(http: Http, collection: str, repo: str, paths: list[str]) -> int:
    """Slet alle punkter for repo+path i spejlet (samme filter som gigakode.py delete_paths)."""
    total = 0
    for i in range(0, len(paths), 100):
        flt = {"must": [{"key": "repo", "match": {"value": repo}},
                        {"key": "path", "match": {"any": paths[i:i + 100]}}]}
        n = http.req("POST", f"/collections/{collection}/points/count", {"filter": flt, "exact": True})
        cnt = int(((n or {}).get("result") or {}).get("count") or 0)
        if cnt:
            http.req("POST", f"/collections/{collection}/points/delete?wait=true", {"filter": flt})
        total += cnt
    return total


def upsert_points(http: Http, collection: str, points: list[dict]) -> int:
    n = 0
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i:i + UPSERT_BATCH]
        http.req("PUT", f"/collections/{collection}/points?wait=true", {"points": batch})
        n += len(batch)
    return n


def mirror_collection(*, source: Http, target: Http, collection: str, repo: str, branch: str,
                      source_repo: str, state_dir: str, dry_run: bool, target_label: str = "") -> dict:
    """Én spejlingskørsel. Returnerer et resultat-dict (til rapport/tests)."""
    src_state = gigakode_state(state_dir, collection, repo)
    src_manifest = read_source_manifest(source, collection) or {}

    commit_sha = str(src_manifest.get("commit_sha") or src_state.get("commit_sha") or "")
    src_files_state: dict = src_state.get("files") or {}

    mirror_state_file = state_path(state_dir, collection, f"{repo}__mirror")
    mirror_state = load_json(mirror_state_file)

    # ---- commit_sha-gate (idempotens) ----
    if commit_sha and mirror_state.get("commit_sha") == commit_sha and mirror_state.get("complete"):
        log_line(event="unchanged", collection=collection, commit_sha=commit_sha, upserts=0, deletes=0)
        return {"status": "unchanged", "commit_sha": commit_sha, "upserts": 0, "deletes": 0,
                "dry_run": dry_run}

    # ---- læs kilden ----
    try:
        source_config = (source.req("GET", f"/collections/{collection}") or {}).get("result") or {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # ingen kilde-collection = intet at spejle. IKKE en tavs "done": en manglende collection
            # er en driftsfejl for kalderen, så vi returnerer `source_missing` og rc=1.
            log_line(event="source_missing", collection=collection,
                     reason="collection findes ikke paa kilden (intet at spejle)")
            return {"status": "source_missing", "commit_sha": commit_sha, "upserts": 0, "deletes": 0,
                    "dry_run": dry_run}
    pts = scroll_all(source, collection)
    expected = source_point_count(source, collection)
    if len(pts) != expected:
        raise RuntimeError(f"scroll gav {len(pts)} punkter, kildens count siger {expected} — afbryder "
                           f"(manglende punkter må ikke blive et tavst tab)")
    if not commit_sha:
        commit_sha = stable_tree_sha([p.get("payload") or {} for p in pts])
        log_line(event="commit_sha_derived", collection=collection, commit_sha=commit_sha, points=len(pts))

    # ---- generation pr. fil (valg dokumenteret i modul-docstringen) ----
    generations: dict[str, str] = {}
    per_file_ok = True
    for p in pts:
        pl = p.get("payload") or {}
        path = str(pl.get("path") or "")
        fh = (src_files_state.get(path) or {}).get("h")
        gen = file_generation(fh)
        if gen is None:
            per_file_ok = False
            break
        generations[path] = gen
    if per_file_ok and generations:
        generation_mode = "state"
    else:
        single = collection_version(repo, branch, commit_sha)
        generations = {str((p.get("payload") or {}).get("path") or ""): single for p in pts}
        generation_mode = "collection"

    log_line(event="read_source", collection=collection, points=len(pts), commit_sha=commit_sha,
             generation_mode=generation_mode, files=len(generations))

    # ---- hvad skal skrives? (pr.-fil sha256-fingeraftryk) ----
    src_fp = source_file_fingerprint([p.get("payload") or {} for p in pts])
    prev_fp = mirror_state.get("files") or {}
    changed = sorted(path for path, fp in src_fp.items() if prev_fp.get(path) != fp)
    removed = sorted(path for path in prev_fp if path not in src_fp)
    to_write = [p for p in pts if str((p.get("payload") or {}).get("path") or "") in set(changed)]

    log_line(event="plan", collection=collection, changed_files=len(changed), removed_files=len(removed),
             points_to_upsert=len(to_write), unchanged_files=len(src_fp) - len(changed))

    if dry_run:
        log_line(event="dry_run", collection=collection, would_upsert=len(to_write),
                 would_delete_files=len(changed) + len(removed), wrote=0)
        return {"status": "dry_run", "commit_sha": commit_sha, "upserts": 0, "deletes": 0,
                "would_upsert": len(to_write), "would_delete_files": len(changed) + len(removed),
                "generation_mode": generation_mode, "dry_run": True}

    # ---- skriv til spejlet ----
    created = ensure_target_collection(target, collection, source_config, target_label)

    deletes = delete_paths(target, collection, repo, changed + removed) if (changed or removed) else 0
    mirror_points = [build_mirror_point(p, branch, generations.get(str((p.get("payload") or {}).get("path") or ""), ""))
                     for p in to_write]
    upserts = upsert_points(target, collection, mirror_points) if mirror_points else 0
    log_line(event="points_written", collection=collection, upserts=upserts, deletes=deletes)

    # ---- manifest-post i helix_branch_manifest @ :16333 ----
    manifest_payload = build_manifest_payload(
        repo=repo, source_repo=source_repo, branch=branch, collection=collection, commit_sha=commit_sha,
        points_count=len(pts), files_count=len(src_fp),
        active_generation={path: gen for path, gen in sorted(generations.items()) if path},
        generation_mode=generation_mode,
        complete=bool(src_manifest.get("complete", True)),
        errors=int(src_manifest.get("errors") or 0),
        mid_ingest=bool(src_manifest.get("mid_ingest", False)),
        canary=KANARIE,
    )
    manifest_payload["b1_namespace"] = "hus"
    mid = manifest_point_id(collection)
    try:
        target.req("GET", f"/collections/{MANIFEST_COLLECTION}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        target.req("PUT", f"/collections/{MANIFEST_COLLECTION}", {"vectors": {"size": 1, "distance": "Dot"}})
    # guard: en ANDEN post må ikke allerede navngive samme collection (helix tager points[0])
    sc = target.req("POST", f"/collections/{MANIFEST_COLLECTION}/points/scroll", {
        "limit": 5, "with_payload": ["collection_name"], "with_vector": False,
        "filter": {"must": [{"key": "collection_name", "match": {"value": collection}}]},
    })
    for p in ((sc or {}).get("result") or {}).get("points") or []:
        if str(p.get("id")) != mid:
            raise GuardError(f"manifest-punkt {p.get('id')} navngiver allerede {collection!r} — afbryder "
                             f"(helix tager points[0], tvetydighed er ikke tilladt)")
    target.req("PUT", f"/collections/{MANIFEST_COLLECTION}/points?wait=true",
               {"points": [{"id": mid, "vector": [0.0], "payload": manifest_payload}]})
    log_line(event="manifest_written", collection=collection, manifest_id=mid, active_generation=len(
        manifest_payload["active_generation"]), generation_mode=generation_mode)

    # ---- gem spejlets egen tilstand ----
    save_json(mirror_state_file, {
        "repo": repo, "collection": collection, "branch": branch, "commit_sha": commit_sha,
        "generation_mode": generation_mode, "complete": bool(manifest_payload["complete"]),
        "files": src_fp, "files_count": len(src_fp), "points_count": len(pts),
        "last_run_ts": int(time.time()), "canary": KANARIE,
    })
    log_line(event="done", collection=collection, commit_sha=commit_sha, upserts=upserts, deletes=deletes)
    return {"status": "written", "commit_sha": commit_sha, "upserts": upserts, "deletes": deletes,
            "generation_mode": generation_mode, "dry_run": False, "collection_created": created}


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spejl code__gigacode__main fra hus-Qdrant :16343 til læse-spejlet :16333")
    p.add_argument("--dry-run", action="store_true", help="læs og planlæg, skriv INTET")
    p.add_argument("--source", default=os.environ.get("GIGACODE_MIRROR_SOURCE", f"http://127.0.0.1:{SOURCE_PORT}"))
    p.add_argument("--target", default=os.environ.get("GIGACODE_MIRROR_TARGET", f"http://127.0.0.1:{TARGET_PORT}"))
    p.add_argument("--collection", default=os.environ.get("GIGAKODE_MIRROR_COLLECTION", "code__gigacode__main"))
    p.add_argument("--state-dir", default=os.environ.get("GIGAKODE_STATE", "/tmp/opencode/gigakode-state"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_url = assert_source(args.source)
        target_url = assert_target(args.target)
        collection = assert_code_collection(args.collection)
        inner = collection[len("code__"):]
        repo, branch = inner.rsplit("__", 1)
        source_repo = repo
        log_line(event="start", collection=collection, source=source_url, target=target_url,
                 dry_run=int(args.dry_run), state_dir=args.state_dir)
        res = mirror_collection(source=Http(source_url), target=Http(target_url), collection=collection,
                                repo=repo, branch=branch, source_repo=source_repo,
                                state_dir=args.state_dir, dry_run=args.dry_run, target_label=target_url)
        log_line(event="ok", collection=collection, status=res.get("status"), commit_sha=res.get("commit_sha"),
                 upserts=res.get("upserts", 0), deletes=res.get("deletes", 0))
        return 1 if res.get("status") == "source_missing" else 0
    except GuardError as exc:
        log_line(event="refused", reason=str(exc))
        return 2
    except urllib.error.HTTPError as exc:
        log_line(event="http_error", code=exc.code, reason=str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 — fejlen skal ses i loggen, ikke skjules
        log_line(event="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
