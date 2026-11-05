"""Tests for GIGACODE-INGEST-20261105 spejl-komponenten (bin/gigacode-mirror.py + .sh + cron).

KANARIE=GIGACODE-INGEST-20261105

Kør (fra /shared/claude-shared/director-v3/sandbox/gigacode-ingest-20261105):
  PYTHONPATH=/home/tlg/.local/lib/python3.12/site-packages python3 -B -m pytest tests/test_gigacode_mirror.py -v -p no:cacheprovider

Alt er offline: INGEN netkald til Qdrant, INGEN skrivning, INGEN cron-installation. HTTP-laget er
erstattet af FakeHttp (nedenfor), der tæller kald og holder et in-memory Qdrant. Syntetiske vaerdier only
(kunstigt repo-navn "demo-repo", kunstige sha'er) — ingen patientdata, ingen ægte repo-navne.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
MIRROR_PY = BASE / "bin" / "gigacode-mirror.py"
MIRROR_SH = BASE / "bin" / "gigacode-mirror.sh"
CRON = BASE / "deploy" / "gigacode-mirror.cron"

sys.dont_write_bytecode = True


def _load_mirror():
    spec = importlib.util.spec_from_file_location("gigacode_mirror", MIRROR_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_mirror()


# --------------------------------------------------------------------------- FakeHttp (offline)
class FakeHttp:
    """In-memory Qdrant. Tæller ALLE kald. Bruges i stedet for m.Http i tests."""

    def __init__(self, collections: dict[str, list[dict]] | None = None, config: dict | None = None):
        self.collections = collections or {}
        self.config = config or {
            "config": {"params": {"vectors": {"size": 2560, "distance": "Cosine"}},
                       "hnsw_config": {"m": 16, "ef_construct": 100}}}
        self.calls: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str]] = []

    def req(self, method: str, path: str, body=None, timeout=None):
        self.calls.append((method, path))
        if method in ("PUT", "POST", "DELETE") and any(
                k in path for k in ("/points", "/collections/")):
            self.writes.append((method, path))
        parts = [p for p in path.split("?")[0].split("/") if p]
        # GET /collections/<name>
        if method == "GET" and len(parts) == 2 and parts[0] == "collections":
            name = parts[1]
            if name not in self.collections and name != m.MANIFEST_COLLECTION:
                raise _http_error(404)
            return {"result": {"config": self.config["config"],
                               "points_count": len(self.collections.get(name, []))}}
        # PUT /collections/<name>
        if method == "PUT" and len(parts) == 2 and parts[0] == "collections":
            name = parts[1]
            self.collections.setdefault(name, [])
            self.collections[name + "__created"] = body  # type: ignore[assignment]
            return {"result": True}
        # POST /collections/<name>/points/count
        if method == "POST" and parts[-1] == "count":
            name = parts[1]
            pts = self.collections.get(name, [])
            flt = (body or {}).get("filter")
            if flt:
                must = flt.get("must") or []
                for c in must:
                    if "any" in (c.get("match") or {}):
                        pts = [p for p in pts if (p.get("payload") or {}).get(c["key"]) in set(c["match"]["any"])]
                        continue
                    pts = [p for p in pts if (p.get("payload") or {}).get(c["key"]) == c["match"]["value"]]
            return {"result": {"count": len(pts)}}
        # POST /collections/<name>/points/scroll
        if method == "POST" and parts[-1] == "scroll":
            name = parts[1]
            pts = list(self.collections.get(name, []))
            flt = (body or {}).get("filter")
            if flt:
                for c in (flt.get("must") or []):
                    key, val = c["key"], c["match"]["value"]
                    pts = [p for p in pts if (p.get("payload") or {}).get(key) == val]
            lim = (body or {}).get("limit", 256)
            off = (body or {}).get("offset") or 0
            page = pts[off:off + lim]
            nxt = off + lim if off + lim < len(pts) else None
            return {"result": {"points": page, "next_page_offset": nxt}}
        # POST /collections/<name>/points/delete
        if method == "POST" and parts[-1] == "delete":
            name = parts[1]
            must = (body or {}).get("filter", {}).get("must", [])
            keep = []
            for p in self.collections.get(name, []):
                pl = p.get("payload") or {}
                hit = True
                for c in must:
                    if c["key"] == "path":
                        hit = hit and pl.get("path") in set(c["match"]["any"])
                    else:
                        hit = hit and pl.get(c["key"]) == c["match"]["value"]
                if not hit:
                    keep.append(p)
            self.collections[name] = keep
            return {"result": {"status": "completed"}}
        # PUT /collections/<name>/points
        if method == "PUT" and parts[-1] == "points":
            name = parts[1]
            cur = self.collections.setdefault(name, [])
            for new in (body or {}).get("points", []):
                cur[:] = [p for p in cur if str(p.get("id")) != str(new.get("id"))]
                cur.append(new)
            return {"result": {"status": "completed"}}
        raise AssertionError(f"FakeHttp: uventet kald {method} {path}")


def _http_error(code):
    import urllib.error
    return urllib.error.HTTPError("fake", code, "fake", {}, None)


def make_points(n: int, repo: str = "demo-repo", branch: str = "main", commit: str = "c0ffee") -> list[dict]:
    out = []
    for i in range(n):
        path = f"src/file{i}.py"
        out.append({
            "id": f"00000000-0000-5000-8000-{i:012d}",
            "vector": [0.01] * 2560,
            "payload": {
                "repo": repo, "branch": branch, "commit_sha": commit, "path": path,
                "line_start": 1, "sha256": hashlib.sha256(f"{path}-v1".encode()).hexdigest(),
                "text": f"def f{i}(): pass",
            },
        })
    return out


def write_gigakode_state(state_dir: Path, collection: str, repo: str, files: dict, commit: str) -> None:
    d = state_dir / collection
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{repo}.json").write_text(json.dumps({
        "repo": repo, "commit_sha": commit, "complete": True,
        "files": files,
    }), encoding="utf-8")


# --------------------------------------------------------------------------- 1. source-guard
def test_source_guard_refuses_16333():
    with pytest.raises(m.GuardError):
        m.assert_source("http://127.0.0.1:16333")


def test_source_guard_accepts_16343():
    assert m.assert_source("http://127.0.0.1:16343") == "http://127.0.0.1:16343"


# --------------------------------------------------------------------------- 2. target-guard
def test_target_guard_refuses_16343():
    with pytest.raises(m.GuardError):
        m.assert_target("http://127.0.0.1:16343")


def test_target_guard_accepts_16333():
    assert m.assert_target("http://127.0.0.1:16333") == "http://127.0.0.1:16333"


# --------------------------------------------------------------------------- 3. target-guard negative
@pytest.mark.parametrize("url", [
    "http://127.0.0.1:6333",
    "http://127.0.0.1:6340",
    "http://127.0.0.1:8700",
    "http://helios.local:16333",
    "http://10.0.0.5:16333",
])
def test_target_guard_refuses_foreign(url):
    with pytest.raises(m.GuardError):
        m.assert_target(url)


# --------------------------------------------------------------------------- 4. --dry-run skriver intet
def test_dry_run_writes_nothing(tmp_path):
    pts = make_points(3)
    src = FakeHttp(collections={"code__demo-repo__main": pts})
    tgt = FakeHttp(collections={})
    res = m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main",
                              repo="demo-repo", branch="main", source_repo="demo-repo",
                              state_dir=str(tmp_path), dry_run=True)
    assert res["status"] == "dry_run"
    assert tgt.writes == [], f"dry-run skrev: {tgt.writes}"
    assert res["upserts"] == 0 and res["deletes"] == 0
    assert not list(tmp_path.rglob("*.json"))


# --------------------------------------------------------------------------- 5. manifest-felter
def test_manifest_payload_has_all_required_fields(tmp_path):
    payload = m.build_manifest_payload(
        repo="demo-repo", source_repo="demo-repo", branch="main",
        collection="code__demo-repo__main", commit_sha="abc123",
        points_count=5, files_count=2, active_generation={"a.py": "0123456789abcdef"},
        generation_mode="state", complete=True, errors=0, mid_ingest=False, canary=m.KANARIE)
    payload["b1_namespace"] = "hus"   # sat af mirror_collection, som main() gør
    for f in m.REQUIRED_MANIFEST_FIELDS:
        assert f in payload, f"mangler manifest-felt {f}"
    assert payload["legacy"] is False
    assert isinstance(payload["active_generation"], dict)
    assert payload["collection_name"] == "code__demo-repo__main"


def test_manifest_required_fields_match_measured_source():
    """Feltlisten skal dække de felter der faktisk står i :16343-manifestet (målt 2026-11-05)."""
    measured = {"repo", "source_repo", "branch", "collection_name", "commit_sha", "points_count",
                "files_count", "updated_at", "legacy", "complete", "errors", "mid_ingest", "canary",
                "b1_namespace"}
    missing = measured - set(m.REQUIRED_MANIFEST_FIELDS)
    assert not missing, f"REQUIRED_MANIFEST_FIELDS mangler: {sorted(missing)}"


# --------------------------------------------------------------------------- 6. generation deterministisk
def test_generation_is_deterministic():
    a = m.collection_version("demo-repo", "main", "abc123")
    b = m.collection_version("demo-repo", "main", "abc123")
    c = m.collection_version("demo-repo", "main", "abc124")
    assert a == b and len(a) == 8
    assert a != c
    assert m.file_generation("0123456789abcdef0123") == "0123456789abcdef"
    assert m.file_generation("short") is None
    assert m.file_generation(None) is None


# --------------------------------------------------------------------------- 7. idempotens: samme sha -> unchanged
def test_idempotent_unchanged_commit_writes_nothing(tmp_path):
    pts = make_points(3)
    src = FakeHttp(collections={"code__demo-repo__main": pts})
    tgt = FakeHttp(collections={})
    # kildens manifest-post bærer commit_sha (målt struktur i :16343)
    src.collections[m.MANIFEST_COLLECTION] = [{
        "id": m.manifest_point_id("code__demo-repo__main"),
        "payload": {"collection_name": "code__demo-repo__main", "commit_sha": "c0ffee", "complete": True},
    }]
    files = {p["payload"]["path"]: {"h": p["payload"]["sha256"]} for p in pts}
    write_gigakode_state(tmp_path, "code__demo-repo__main", "demo-repo", files, "c0ffee")
    # første kørsel skriver
    m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main", repo="demo-repo",
                        branch="main", source_repo="demo-repo", state_dir=str(tmp_path), dry_run=False)
    tgt.writes.clear()
    src.calls.clear()
    # anden kørsel: samme commit -> unchanged, 0 skriv
    res = m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main", repo="demo-repo",
                              branch="main", source_repo="demo-repo", state_dir=str(tmp_path), dry_run=False)
    assert res["status"] == "unchanged"
    assert res["upserts"] == 0 and res["deletes"] == 0
    assert tgt.writes == [], f"unchanged skrev: {tgt.writes}"
    # ingen scroll/count mod kilden heller (kun manifest-scroll til sha-gaten)
    assert not any("/points/scroll" in p and "manifest" not in p for _, p in src.calls)


# --------------------------------------------------------------------------- 8. ændret sha256 -> kun ændrede
def test_changed_sha256_upserts_only_changed_points(tmp_path):
    pts = make_points(3)
    src = FakeHttp(collections={"code__demo-repo__main": pts})
    tgt = FakeHttp(collections={})
    src.collections[m.MANIFEST_COLLECTION] = [{
        "id": m.manifest_point_id("code__demo-repo__main"),
        "payload": {"collection_name": "code__demo-repo__main", "commit_sha": "c0ffee", "complete": True},
    }]
    files = {p["payload"]["path"]: {"h": p["payload"]["sha256"]} for p in pts}
    write_gigakode_state(tmp_path, "code__demo-repo__main", "demo-repo", files, "c0ffee")
    m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main", repo="demo-repo",
                        branch="main", source_repo="demo-repo", state_dir=str(tmp_path), dry_run=False)
    # ændr file1 (1 af 3) og bump commit
    new_sha = hashlib.sha256(b"src/file1.py-v2").hexdigest()
    src.collections["code__demo-repo__main"][1]["payload"]["sha256"] = new_sha
    src.collections[m.MANIFEST_COLLECTION][0]["payload"]["commit_sha"] = "c0ffee2"
    files["src/file1.py"] = {"h": new_sha}
    write_gigakode_state(tmp_path, "code__demo-repo__main", "demo-repo", files, "c0ffee2")
    tgt.writes.clear()
    res = m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main", repo="demo-repo",
                              branch="main", source_repo="demo-repo", state_dir=str(tmp_path), dry_run=False)
    assert res["status"] == "written"
    assert res["upserts"] == 1, f"upsertede {res['upserts']} punkter, forventet 1"
    upsert_calls = [p for _, p in tgt.writes if p.endswith("/points?wait=true") and "manifest" not in p]
    assert len(upsert_calls) == 1
    mirrored = {p["payload"]["path"] for p in tgt.collections["code__demo-repo__main"]}
    assert mirrored == {"src/file0.py", "src/file1.py", "src/file2.py"}


# --------------------------------------------------------------------------- 9. code__-præfiks håndhæves
def test_code_prefix_is_enforced():
    with pytest.raises(m.GuardError):
        m.assert_code_collection("gigacode")
    with pytest.raises(m.GuardError):
        m.assert_code_collection("github_rag")
    assert m.assert_code_collection("code__gigacode__main") == "code__gigacode__main"


# --------------------------------------------------------------------------- 10. collection-config fra kilden
def test_collection_config_copied_from_source():
    src_cfg = {"config": {"params": {"vectors": {"size": 2560, "distance": "Cosine"}},
                          "hnsw_config": {"m": 16, "ef_construct": 100}}}
    body = m.collection_create_body(src_cfg)
    assert body["vectors"]["size"] == 2560
    assert body["vectors"]["distance"] == "Cosine"
    assert body["on_disk_payload"] is True
    assert body["hnsw_config"]["m"] == 16
    # negativ: forkert dim afvises (frozen 2560)
    with pytest.raises(m.GuardError):
        m.collection_create_body({"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}})


# --------------------------------------------------------------------------- 11. flock-linje + bash -n
def test_wrapper_syntax_and_flock_line():
    assert MIRROR_SH.exists(), f"mangler {MIRROR_SH}"
    text = MIRROR_SH.read_text(encoding="utf-8")
    assert "flock -n 9 || exit 0" in text
    assert "set -uo pipefail" in text
    assert "/tmp/gigakode-locks" in text
    assert "KANARIE=GIGACODE-INGEST-20261105" in text
    r = subprocess.run(["bash", "-n", str(MIRROR_SH)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n fejlede: {r.stderr}"


# --------------------------------------------------------------------------- 12. kanarie i hver loglinje
def test_every_log_line_has_canary(tmp_path, monkeypatch):
    log = tmp_path / "mirror.log"
    monkeypatch.setenv("GIGAKODE_MIRROR_LOG", str(log))
    monkeypatch.setattr(m, "LOG_PATH", str(log))
    m.log_line(event="test", collection="code__demo-repo__main")
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert lines and all("KANARIE=GIGACODE-INGEST-20261105" in ln for ln in lines)


# --------------------------------------------------------------------------- 13. cron staged
def test_cron_is_staged_with_one_active_line():
    assert CRON.exists(), f"mangler {CRON}"
    lines = CRON.read_text(encoding="utf-8").splitlines()
    active = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    assert len(active) == 1, f"forventet 1 aktiv linje, fandt {len(active)}"
    assert active[0].startswith("*/5 * * * *")
    assert "bin/gigacode-mirror.sh" in active[0]
    assert ">> /tmp/gigakode-mirror.log 2>&1" in active[0]
    assert any("Tommy" in ln for ln in lines), "mangler kommentar om Tommy-GO"


# --------------------------------------------------------------------------- 14. CLI-guards via main()
def test_main_refuses_bad_source_and_target():
    assert m.main(["--source", "http://127.0.0.1:16333"]) == 2
    assert m.main(["--target", "http://127.0.0.1:16343"]) == 2
    assert m.main(["--collection", "gigacode"]) == 2


# --------------------------------------------------------------------------- 15. manifest-id = ingest-namespace
def test_manifest_point_id_matches_ingest_convention():
    import uuid
    expected = str(uuid.uuid5(uuid.UUID("2f8d4c61-7a3e-5b90-8d14-6c2fa9e07b53"), "demo-repo:main"))
    assert m.manifest_point_id("code__demo-repo__main") == expected
    with pytest.raises(m.GuardError):
        m.manifest_point_id("code__nounderscore")


# --------------------------------------------------------------------------- 16. gate-felter på punkterne
def test_mirror_points_carry_generation_branch_source_path():
    pt = make_points(1)[0]
    out = m.build_mirror_point(pt, "main", "0123456789abcdef")
    assert out["payload"]["generation"] == "0123456789abcdef"
    assert out["payload"]["branch"] == "main"
    assert out["payload"]["source_path"] == out["payload"]["path"] == "src/file0.py"
    assert out["id"] == pt["id"] and len(out["vector"]) == 2560
    assert out["payload"]["sha256"] == pt["payload"]["sha256"]  # 1:1 kopi, intet tabt


# --------------------------------------------------------------------------- 17. kilde mangler -> source_missing
def test_missing_source_collection_is_reported_not_silent(tmp_path):
    src = FakeHttp(collections={})   # ingen code__demo-repo__main
    tgt = FakeHttp(collections={})
    res = m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main", repo="demo-repo",
                              branch="main", source_repo="demo-repo", state_dir=str(tmp_path), dry_run=False)
    assert res["status"] == "source_missing"
    assert tgt.writes == [], f"skrev selvom kilden mangler: {tgt.writes}"


# --------------------------------------------------------------------------- 18. scroll-tælling != count afbryder
def test_scroll_count_mismatch_aborts(tmp_path):
    class ShortScroll(FakeHttp):
        def req(self, method, path, body=None, timeout=None):
            if method == "POST" and path.endswith("/points/count") and "manifest" not in path:
                return {"result": {"count": 99}}
            return super().req(method, path, body, timeout)

    pts = make_points(3)
    src = ShortScroll(collections={"code__demo-repo__main": pts})
    tgt = FakeHttp(collections={})
    with pytest.raises(RuntimeError):
        m.mirror_collection(source=src, target=tgt, collection="code__demo-repo__main", repo="demo-repo",
                            branch="main", source_repo="demo-repo", state_dir=str(tmp_path), dry_run=False)
