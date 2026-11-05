#!/usr/bin/env python3
"""gigakode.py - GIGAKODE: one living code-RAG collection ("legoklodser" for LOVELACE/Helix).

KANARIE=GIGAKODE-20261103   (Tommy's order 2026-11-03 23:23-23:30)

One Qdrant collection `gigakode` (2560d cosine, vectors on_disk) on the house Qdrant (127.0.0.1:16343).
Isolation is a payload filter (repo, branch, kilde, sprog, element_type, path, line_start) - exactly like
patient_lnr in patient_journal_rag. One point = one function/class/route/config-block/doc-section.

Reuses code_rag.py (READ ONLY, imported, never copied/edited): chunk_file + the language chunkers,
chunker_version, mask_secrets, walk_files/file_allowed (file selection), Embedder (LiteLLM embed-qwen3),
repo_branch. Repo list = the KODE-RAG registry (etc/repos-b1.json + etc/repos.json); a new repo in the
registry is picked up by the next run.

Living / incremental: per repo the last indexed commit_sha lives in state/<collection>/<repo>.json.
New sha -> `git diff` narrows the candidate files, content hash decides, changed files have their points
deleted (filter repo+path) and re-inserted. Point ids are uuid5(repo|branch|path|line_start|chunker_version),
so a repeated run is idempotent.

CLI:
  gigakode.py sync   [--repo X ...] [--dry-run] [--max-files N] [--full]
  gigakode.py status [--no-qdrant]                 (JSON on stdout - the UI reads this)
Common: --collection NAME (env GIGAKODE_COLLECTION, default gigakode) --qdrant URL (env GIGAKODE_QDRANT)
        --registry FILE[,FILE] (env GIGAKODE_REGISTRY) --state-dir DIR (env GIGAKODE_STATE)
Env: GIGAKODE_EMBEDDER=fake  -> deterministic fake vectors (tests only, never production).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import inspect
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the shared code-rag/lib directory
CODE_RAG_LIB = os.environ.get("GIGAKODE_CODE_RAG_LIB", "/shared/claude-shared/code-rag/lib")
sys.path.insert(0, CODE_RAG_LIB)
import code_rag as cr  # noqa: E402  (READ ONLY reuse)

KANARIE = "GIGAKODE-20261103"
SANDBOX = Path(__file__).resolve().parent.parent
CODE_RAG_BASE = Path(CODE_RAG_LIB).parent
DEFAULT_COLLECTION = "gigakode"
DEFAULT_QDRANT = "http://127.0.0.1:16343"   # hus-qdrant on the host
VECTOR_SIZE = 2560                          # FROZEN (embed-qwen3). Never change model/dimension.
DEFAULT_REGISTRIES = [CODE_RAG_BASE / "etc" / "repos-b1.json", CODE_RAG_BASE / "etc" / "repos.json"]
B1_STATE_DIR = CODE_RAG_BASE / "state" / "b1"
EMBED_BATCH = 16
EMBED_WORKERS = 4
GROUP_FILES = 40            # files per delete+embed+upsert group (state is saved after every group)
GROUP_CHUNKS = 160
MAX_TEXT_CHARS = cr.MAX_CHUNK_CHARS
ELEMENT_TYPES = ("function", "class", "method", "route", "config", "doc", "test", "other")
KILDER = ("hus", "ekstern", "projekt", "hippocampus")
KEYWORD_FIELDS = ("repo", "branch", "kilde", "sprog", "element_type", "path")
INTEGER_FIELDS = ("line_start",)
LOG_PATH = Path(os.environ.get("GIGAKODE_LOG", SANDBOX / "logs" / "gigakode.log"))


# --------------------------------------------------------------------------- log
def ts_iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else time.time()))


def log_line(**kv) -> None:
    """Every log line carries KANARIE=GIGAKODE-20261103."""
    parts = [ts_iso(), f"KANARIE={KANARIE}"] + [f"{k}={str(v).replace(chr(10), ' ')}" for k, v in kv.items()]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(" ".join(parts) + "\n")


# --------------------------------------------------------------------------- registry
def load_registry(spec: str | None) -> list[dict]:
    """Merge registry files (first definition of a name wins). Default: repos-b1.json + repos.json."""
    paths = [Path(p) for p in spec.split(",") if p] if spec else DEFAULT_REGISTRIES
    seen: dict[str, dict] = {}
    for p in paths:
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for e in data.get("repos", []):
            if e.get("name") and e["name"] not in seen:
                seen[e["name"]] = e
    return list(seen.values())


def kilde_of(entry: dict) -> str:
    """hus | ekstern | projekt | hippocampus. Registry override: entry['gigakode_kilde']."""
    k = entry.get("gigakode_kilde")
    if k in KILDER:
        return k
    name = entry.get("name", "")
    if entry.get("kind") == "ekstern" or entry.get("b1_target") == "ekstern":
        return "ekstern"
    if "hippocampus" in name.lower():
        return "hippocampus"
    if entry.get("b1_target") == "hus" or name.startswith("hus__"):
        return "hus"
    return "projekt"


# --------------------------------------------------------------------------- classification
SPROG_BY_SUFFIX = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".sh": "shell", ".bash": "shell", ".yaml": "yaml",
    ".yml": "yaml", ".md": "markdown", ".html": "html", ".json": "json",
}
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|testdata|fixtures?)/|"
    r"(^|/)(test_[^/]*\.py|[^/]*_test\.(py|js|ts)|conftest\.py|[^/]*\.(test|spec)\.[jt]sx?)$",
    re.I,
)
ROUTE_PY_RE = re.compile(r"^\s*@\s*[\w.]+\.(get|post|put|delete|patch|head|options|route|websocket|api_route)\s*\(", re.I)
ROUTE_NEST_RE = re.compile(r"^\s*@\s*(Get|Post|Put|Delete|Patch|All)\s*\(")
ROUTE_JS_RE = re.compile(
    r"\b(app|router|server|api|fastify|route|routes)\s*\.\s*(get|post|put|delete|patch|all|options|head)\s*\(\s*[`'\"]"
)
PART_RE = re.compile(r"#(\d+)(?:~\d+)?$")
BASE_RE = re.compile(r"(?:#\d+)?(?:~\d+)?$")
DEF_RE = re.compile(r"^(\s*)(async\s+def|def|class)\s+")
JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+")


def _decorators_above(lines: list[str], start: int) -> list[str]:
    out: list[str] = []
    i = start - 2
    while i >= 0 and lines[i].lstrip().startswith("@"):
        out.append(lines[i].strip())
        i -= 1
    return list(reversed(out))


def _kind_of_first_line(suffix: str, rel: str, lines: list[str], start: int) -> tuple[str, str]:
    """(element_type, route_hint) from the chunk's first line + decorators above it (part 1 only)."""
    first = lines[start - 1] if 0 < start <= len(lines) else ""
    lang = cr.LANGS.get(suffix)
    deco = _decorators_above(lines, start)
    if lang == "python":
        for d in deco:
            if ROUTE_PY_RE.match(d) or ROUTE_NEST_RE.match(d):
                return "route", d[:200]
        m = DEF_RE.match(first)
        if m and m.group(2) == "class":
            return "class", ""
        if m:
            return ("method" if len(m.group(1)) > 0 else "function"), ""
        return "other", ""
    if lang == "js":
        for d in deco:
            if ROUTE_NEST_RE.match(d):
                return "route", d[:200]
        if ROUTE_JS_RE.search(first):
            return "route", first.strip()[:200]
        if JS_CLASS_RE.match(first):
            return "class", ""
        indent = len(first) - len(first.lstrip())
        if re.match(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b", first) or re.match(
            r"^\s*(?:export\s+)?(?:const|let|var)\b", first
        ):
            return "function", ""
        return ("method" if indent > 0 else "function"), ""
    if lang == "shell":
        return "function", ""
    if lang in ("yaml", "json"):
        return "config", ""
    if lang == "markdown":
        return "doc", ""
    if lang == "html":
        return ("function" if re.match(r"^\s*(?:async\s+)?function\b|^\s*(?:const|let|var)\s", first) else "other"), ""
    return "other", ""


ROUTE_JS_DETAIL_RE = re.compile(
    r"\b(?:app|router|server|api|fastify|route|routes)\s*\.\s*(get|post|put|delete|patch|all|options|head)\s*\(\s*[`'\"]([^`'\"]*)"
)
_STR_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`")


def extra_js_route_chunks(lines: list[str], existing: list) -> list:
    """code_rag's JS chunker has no pattern for `app.get('/x', (req, res) => {...})`, so Express/Fastify routes would only
    live in the tail of the previous chunk. Add one extra (overlapping, like class/method) chunk per route statement:
    from the route line to where its parentheses balance (capped at code_rag.MAX_CHUNK_LINES)."""
    starts = {c.start for c in existing}
    out = []
    for i, line in enumerate(lines):
        if (i + 1) in starts or len(line) - len(line.lstrip()) > 4:
            continue
        m = ROUTE_JS_DETAIL_RE.search(line)
        if not m:
            continue
        bal = 0
        end = i
        for j in range(i, min(len(lines), i + cr.MAX_CHUNK_LINES)):
            code = _STR_RE.sub('""', lines[j])
            bal += code.count("(") - code.count(")")
            end = j
            if bal <= 0:
                break
        sym = f"{m.group(1).upper()} {m.group(2)}"[:80]
        out.append(cr.Chunk(symbol=sym, start=i + 1, end=end + 1, text="\n".join(lines[i:end + 1])))
    return out


def classify_chunks(rel: str, suffix: str, lines: list[str], chunks: list) -> list[dict]:
    """One dict per chunk: element_type, route, signature, symbol (clean name), part."""
    is_test = bool(TEST_PATH_RE.search(rel))
    out: list[dict] = []
    last_kind: tuple[str, str] | None = None
    for c in chunks:
        m = PART_RE.search(c.symbol)
        part = int(m.group(1)) if m else 1
        base = BASE_RE.sub("", c.symbol)
        if c.symbol.startswith("window-"):
            kind, route = "other", ""
            last_kind = None
        elif part == 1 or last_kind is None:
            kind, route = _kind_of_first_line(suffix, rel, lines, c.start)
            last_kind = (kind, route)
        else:
            kind, route = last_kind          # continuation part of a split chunk inherits part 1's kind
        if is_test:
            kind = "test"
        sig = ""
        if part == 1:
            for ln in c.text.splitlines():
                if ln.strip():
                    sig = ln.strip()[:200]
                    break
        out.append({"element_type": kind, "route": route, "signature": sig, "symbol": base, "part": part})
    return out


# --------------------------------------------------------------------------- masking fingerprint
def mask_fingerprint() -> str:
    """Changes whenever code_rag's masking rules change -> files are re-chunked on the next run (old points
    may contain values the new rules mask)."""
    h = hashlib.sha256()
    for name in ("MASK_PATTERNS", "LONG_SECRET_RE", "LNR_RE", "CPR_RE", "EMAIL_RE", "PEM_RE"):
        obj = getattr(cr, name, None)
        if obj is None:
            continue
        if isinstance(obj, (list, tuple)):
            h.update("\x00".join(p.pattern for p in obj).encode())
        else:
            h.update(obj.pattern.encode())
    try:
        h.update(inspect.getsource(cr.mask_secrets).encode())
    except (OSError, TypeError):
        pass
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- git helpers
def git(root: str | Path, *args: str, timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception:
        return 1, ""


def stable_tree_sha(files: dict[str, dict]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\x00{files[rel]['h']}\n".encode())
    return "nogit-" + h.hexdigest()[:12]


def git_facts(entry: dict, root: Path) -> dict:
    """{is_git, head, clean, trusted_clean, branch}. trusted_clean only for GitHub mirror clones (reset --hard
    + clean by b1_gh_sync): there HEAD == the working tree, so the sha gate and `git diff` are exact."""
    gr = entry.get("git_root")
    cfg_branch = entry.get("branch", "live")
    if not gr or not (Path(gr) / ".git").exists():
        return {"is_git": False, "head": "", "clean": False, "trusted_clean": False, "branch": cfg_branch}
    rc, head = git(root, "rev-parse", "HEAD")
    head = head.strip() if rc == 0 else ""
    branch = cfg_branch
    if entry.get("branch_from_git"):
        b = cr.repo_branch(entry)
        branch = b if b and b != "HEAD" else cfg_branch
    rc, porc = git(root, "status", "--porcelain", "--", ".", timeout=300)
    clean = rc == 0 and porc.strip() == ""
    return {
        "is_git": bool(head), "head": head, "clean": clean,
        "trusted_clean": bool(entry.get("github")) and clean and bool(head), "branch": branch,
    }


def b1_gate(entry: dict, head: str) -> tuple[bool, str]:
    """Fail-closed on the B1 scan verdict (state/b1/<name>.sync.json): RED/ERROR or a blocked sha = no embed."""
    p = Path(os.environ.get("GIGAKODE_B1_STATE", B1_STATE_DIR)) / f"{entry['name']}.sync.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, ""
    if d.get("scan") in ("RED", "ERROR"):
        return False, f"b1-scan {d.get('scan')}"
    if d.get("blocked_sha") and (not head or d.get("blocked_sha") == head):
        return False, "b1 blocked_sha"
    return True, d.get("scan", "") or ""


def entry_fingerprint(entry: dict, branch: str) -> str:
    keys = ("root", "git_root", "include", "exclude", "scan_exclude_paths", "gigakode_kilde", "kind", "b1_target")
    blob = json.dumps({k: entry.get(k) for k in keys}, sort_keys=True) + f"|{branch}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- qdrant
FORBIDDEN_PORTS = (6333, 6337, 6340, 8700)     # Helios / system Qdrant: Hippocampus must NEVER touch them (Tommy 2026-11-03 22:44 UTC)


def assert_hus_qdrant(url: str) -> str:
    """HARD RULE: gigakode talks to hus-Qdrant http://127.0.0.1:16343 only. Anything else is refused before any request."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if "helios" in (u.hostname or "").lower() or u.port in FORBIDDEN_PORTS or u.hostname not in ("127.0.0.1", "localhost") or u.port != 16343:
        raise SystemExit(f"gigakode: afvist Qdrant-URL {url!r} - kun hus-Qdrant http://127.0.0.1:16343 er tilladt (Helios-forbud)")
    return url.rstrip("/")


class Qdrant:
    def __init__(self, url: str):
        self.url = assert_hus_qdrant(url)

    def req(self, method: str, path: str, body=None, timeout: int = 180):
        return cr._request(method, f"{self.url}{path}", body, timeout=timeout)[1]

    def collection(self, name: str) -> dict | None:
        try:
            return self.req("GET", f"/collections/{name}", timeout=30).get("result", {})
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def ensure(self, name: str) -> bool:
        """Create collection + payload indexes if missing. 2560d cosine, vectors/hnsw/payload on disk."""
        info = self.collection(name)
        created = False
        if info is None:
            self.req("PUT", f"/collections/{name}", {
                "vectors": {"size": VECTOR_SIZE, "distance": "Cosine", "on_disk": True},
                "hnsw_config": {"on_disk": True},
                "on_disk_payload": True,
            })
            created = True
            info = self.collection(name) or {}
        else:
            vec = info.get("config", {}).get("params", {}).get("vectors", {})
            size = vec.get("size") if isinstance(vec, dict) else None
            if size and size != VECTOR_SIZE:
                raise RuntimeError(f"{name}: vector size {size} != {VECTOR_SIZE} (frozen)")
        have = set((info.get("payload_schema") or {}).keys())
        for f in KEYWORD_FIELDS:
            if f not in have:
                self.req("PUT", f"/collections/{name}/index?wait=true",
                         {"field_name": f, "field_schema": {"type": "keyword", "on_disk": True}})
        for f in INTEGER_FIELDS:
            if f not in have:
                self.req("PUT", f"/collections/{name}/index?wait=true",
                         {"field_name": f, "field_schema": {"type": "integer", "lookup": True, "range": True, "on_disk": True}})
        return created

    def delete_paths(self, name: str, repo: str, paths: list[str]) -> int:
        """Delete all points of repo+path(s) (any branch). Returns the number of points that matched."""
        total = 0
        for i in range(0, len(paths), 100):
            flt = {"must": [{"key": "repo", "match": {"value": repo}},
                            {"key": "path", "match": {"any": paths[i:i + 100]}}]}
            n = self.req("POST", f"/collections/{name}/points/count", {"filter": flt, "exact": True}).get("result", {}).get("count", 0)
            if n:
                self.req("POST", f"/collections/{name}/points/delete?wait=true", {"filter": flt})
            total += n
        return total

    def upsert(self, name: str, points: list[dict]) -> None:
        self.req("PUT", f"/collections/{name}/points?wait=true", {"points": points})

    def facet(self, name: str, key: str) -> dict[str, int]:
        res = self.req("POST", f"/collections/{name}/facet", {"key": key, "limit": 2000, "exact": True}, timeout=120)
        return {str(h["value"]): h["count"] for h in res.get("result", {}).get("hits", [])}


# --------------------------------------------------------------------------- embedder
class FakeEmbedder:
    """Deterministic 2560d unit vectors from the text hash. Tests only (GIGAKODE_EMBEDDER=fake)."""
    tokens = 0
    calls = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            rnd = random.Random(int(hashlib.sha256(t.encode()).hexdigest()[:16], 16))
            v = [rnd.gauss(0, 1) for _ in range(VECTOR_SIZE)]
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        self.calls += 1
        return out


def make_embedder():
    if os.environ.get("GIGAKODE_EMBEDDER") == "fake":
        return FakeEmbedder()
    if os.environ.get("GIGAKODE_EMBED_URL"):
        cr.EMBED_URL = os.environ["GIGAKODE_EMBED_URL"]
    return cr.Embedder()


def embed_with_retry(embedder, texts: list[str], tries: int = 3) -> list[list[float]]:
    last: Exception | None = None
    for i in range(tries):
        try:
            vecs = embedder.embed_batch(texts)
            if len(vecs) != len(texts) or any(len(v) != VECTOR_SIZE for v in vecs):
                raise RuntimeError(f"embed gav {len(vecs)} vektorer / dim {len(vecs[0]) if vecs else 0} for {len(texts)} tekster")
            return vecs
        except Exception as exc:  # noqa: BLE001 - retry on any transport/5xx error
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"embedding fejlede efter {tries} forsoeg: {type(last).__name__}: {str(last)[:160]}")


# --------------------------------------------------------------------------- state
def state_dir(args, collection: str) -> Path:
    root = Path(args.state_dir or os.environ.get("GIGAKODE_STATE") or SANDBOX / "state")
    return root / collection


def safe_name(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", repo)


def load_state(sd: Path, repo: str) -> dict:
    p = sd / f"{safe_name(repo)}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(sd: Path, repo: str, st: dict) -> None:
    sd.mkdir(parents=True, exist_ok=True)
    p = sd / f"{safe_name(repo)}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


class RunLock:
    def __init__(self, sd: Path):
        sd.mkdir(parents=True, exist_ok=True)
        self.fh = open(sd / ".lock", "a+")

    def __enter__(self):
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit("en anden gigakode-sync skriver allerede denne collection (lock)")
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        self.fh.close()


# --------------------------------------------------------------------------- file -> points
def point_id(repo: str, branch: str, rel: str, line_start: int, cv: int, salt: str = "") -> str:
    """uuid5(repo|branch|path|line_start|chunker_version). `salt` (symbol) is only added when two chunks of one file start on
    the same line (a split part of a big chunk landing on another chunk's first line, measured on faster-whisper README.md:63)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"gigakode|{repo}|{branch}|{rel}|{line_start}|{cv}" + (f"|{salt}" if salt else "")))


def build_points(entry: dict, root: Path, rel: str, kilde: str, branch: str, commit: str) -> dict:
    """Chunk one file into legoklodser. Returns {points:[{id,embed_text,payload}], types:Counter, masked:int, cv, sprog}."""
    path = root / rel
    suffix = path.suffix.lower()
    lang = cr.LANGS.get(suffix)
    sprog = SPROG_BY_SUFFIX.get(suffix, lang or "unknown")
    cv = cr.chunker_version(lang)
    chunks = cr.chunk_file(path)
    res = {"points": [], "types": collections.Counter(), "masked": 0, "cv": cv, "sprog": sprog}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return res
    if lang == "js" and lines and "\x00" not in "\n".join(lines[:5]):
        chunks = chunks + extra_js_route_chunks(lines, chunks)
        chunks.sort(key=lambda c: (c.start, -c.end))
    if not chunks:
        return res
    info = classify_chunks(rel, suffix, lines, chunks)
    repo = entry["name"]
    display = entry.get("repo_display") or repo
    now = int(time.time())
    used_ids: set[str] = set()
    for c, ci in zip(chunks, info):
        masked = cr.mask_secrets(c.text)               # BEFORE payload AND embed
        if not masked.strip():
            continue
        masked = masked[:MAX_TEXT_CHARS]
        if masked != c.text[:MAX_TEXT_CHARS]:
            res["masked"] += 1
        payload = {
            "repo": repo, "repo_display": display, "branch": branch, "commit_sha": commit, "path": rel,
            "line_start": c.start, "line_end": c.end, "element_type": ci["element_type"],
            "symbol": ci["symbol"], "signature": ci["signature"], "part": ci["part"],
            "sprog": sprog, "kilde": kilde, "text": masked, "chunker_version": cv,
            "sha256": hashlib.sha256(masked.encode("utf-8")).hexdigest(),
            "indekseret_ts": now, "source_pipeline": "gigakode",
        }
        if ci["route"]:
            payload["route"] = cr.mask_secrets(ci["route"])
        header = cr.mask_secrets(f"[{kilde}] {display} {rel}:{c.start} {ci['element_type']} {ci['symbol']}")
        pid = point_id(repo, branch, rel, c.start, cv)
        if pid in used_ids:                                   # same start line as an earlier chunk of this file
            pid = point_id(repo, branch, rel, c.start, cv, f"{ci['symbol']}#{ci['part']}:{c.end}")
        used_ids.add(pid)
        res["points"].append({
            "id": pid,
            "embed_text": (header + "\n" + masked)[: MAX_TEXT_CHARS + 300],
            "payload": payload,
        })
        res["types"][ci["element_type"]] += 1
    return res


def file_hash(path: Path) -> tuple[str, int] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest(), len(data)


# --------------------------------------------------------------------------- sync
def plan_candidates(entry, root, facts, st, allowed: set[str], mask_fp: str) -> tuple[list[str], str]:
    """(candidate rel paths, mode). mode 'diff' = narrowed by git diff, 'scan' = every allowed file."""
    old = st.get("commit_sha") or ""
    prev_files = st.get("files", {})
    if (facts["trusted_clean"] and old and st.get("complete") and st.get("entry_fp") == entry_fingerprint(entry, facts["branch"])
            and old != facts["head"]):
        rc, _ = git(root, "cat-file", "-e", f"{old}^{{commit}}")
        if rc == 0:
            rc, out = git(root, "diff", "--name-status", "--no-renames", "-z", "--relative", old, facts["head"], timeout=300)
            if rc == 0:
                toks = out.split("\0")
                changed = set()
                for i in range(0, len(toks) - 1, 2):
                    if toks[i][:1] in ("A", "M", "T", "C"):
                        changed.add(toks[i + 1])
                cand = set(changed) & allowed
                cand |= allowed - set(prev_files)                                   # new / previously excluded files
                cand |= {r for r, f in prev_files.items() if r in allowed and (f.get("mv") != mask_fp)}
                cand |= {r for r in allowed if r in prev_files and prev_files[r].get("cv") != cr.chunker_version(cr.LANGS.get(Path(r).suffix.lower()))}
                return sorted(cand), "diff"
    return sorted(allowed), "scan"


def run_repo(entry: dict, args, collection: str, q: Qdrant | None, embedder, sd: Path, mask_fp: str) -> dict:
    t0 = time.time()
    name = entry["name"]
    kilde = kilde_of(entry)
    rep: dict = {"repo": name, "kilde": kilde, "status": "", "dry_run": bool(args.dry_run)}
    if not entry.get("enabled", True):
        rep.update(status="skipped", reason=entry.get("disabled_reason") or "enabled=false")
        return rep
    root = Path(entry.get("root", ""))
    if not entry.get("root") or not root.is_dir():
        rep.update(status="skipped", reason="root mangler paa disk (klon ikke synket endnu)")
        return rep
    facts = git_facts(entry, root)
    branch, head = facts["branch"], facts["head"]
    rep["branch"] = branch
    ok, why = b1_gate(entry, head)
    if not ok:
        rep.update(status="blocked", reason=why)
        return rep

    st = load_state(sd, name)
    fp = entry_fingerprint(entry, branch)
    prev_files: dict = st.get("files", {})
    scan_ex = set(entry.get("scan_exclude_paths") or [])
    allowed_paths = [p for p in cr.walk_files(entry)]
    allowed = {str(p.relative_to(root)) for p in allowed_paths} - scan_ex
    rep["files_allowed"] = len(allowed)

    # ---- sha gate ----
    if (not args.full and facts["trusted_clean"] and head and st.get("commit_sha") == head and st.get("complete")
            and st.get("entry_fp") == fp and all(f.get("mv") == mask_fp for f in prev_files.values())):
        rep.update(status="unchanged", commit_sha=head, files_to_process=0, files_deleted=0, chunks=0)
        rep["seconds"] = round(time.time() - t0, 2)
        return rep

    cand, mode = plan_candidates(entry, root, facts, st, allowed, mask_fp)
    rep["mode"] = mode
    todo: list[tuple[str, str, int]] = []
    capped = False
    for rel in cand:
        hs = file_hash(root / rel)
        if hs is None:
            continue
        h, size = hs
        pf = prev_files.get(rel)
        cv = cr.chunker_version(cr.LANGS.get(Path(rel).suffix.lower()))
        if (not args.full and pf and pf.get("h") == h and pf.get("cv") == cv and pf.get("mv") == mask_fp
                and pf.get("branch", branch) == branch):
            continue
        if args.max_files and len(todo) >= args.max_files:
            capped = True
            break
        todo.append((rel, h, size))
    deleted = sorted(set(prev_files) - allowed)
    rep["files_to_process"] = len(todo)
    rep["files_deleted"] = len(deleted)
    rep["capped"] = capped

    if not args.dry_run:
        assert q is not None
        q.ensure(collection)

    commit = head or ""
    tree_files = {r: {"h": h} for r, h, _ in todo}
    if not facts["is_git"]:
        merged = {r: {"h": f["h"]} for r, f in prev_files.items() if r in allowed and r not in tree_files}
        merged.update(tree_files)
        commit = stable_tree_sha(merged) if (not capped and len(merged) == len(allowed)) else (st.get("commit_sha") or "")

    types: collections.Counter = collections.Counter()
    sprogs: collections.Counter = collections.Counter()
    n_chunks = masked_n = 0
    n_deleted_pts = 0
    new_files = {r: f for r, f in prev_files.items() if r in allowed}
    errors: list[str] = []

    def save(complete: bool) -> None:
        st2 = {
            "repo": name, "kilde": kilde, "branch": branch, "root": str(root), "entry_fp": fp,
            "commit_sha": (commit if complete else st.get("commit_sha", "")),
            "pending_sha": ("" if complete else (head or commit)),
            "complete": complete, "trusted_clean": facts["trusted_clean"],
            "collection": collection, "last_run_ts": int(time.time()), "last_run_iso": ts_iso(),
            "files": new_files,
            "totals": {"files": len(new_files), "chunks": sum(f.get("n", 0) for f in new_files.values())},
        }
        save_state(sd, name, st2)

    # deletions of files that vanished / are now excluded
    if deleted and not args.dry_run:
        n_deleted_pts += q.delete_paths(collection, name, deleted)
    for r in deleted:
        new_files.pop(r, None)

    i = 0
    try:
        while i < len(todo):
            group: list[tuple[str, str, int, dict]] = []
            gchunks = 0
            while i < len(todo) and len(group) < GROUP_FILES and gchunks < GROUP_CHUNKS:
                rel, h, size = todo[i]
                b = build_points(entry, root, rel, kilde, branch, commit or head)
                group.append((rel, h, size, b))
                gchunks += len(b["points"])
                i += 1
            for rel, h, size, b in group:
                types.update(b["types"])
                sprogs[b["sprog"]] += len(b["points"])
                n_chunks += len(b["points"])
                masked_n += b["masked"]
            if args.dry_run:
                continue
            n_deleted_pts += q.delete_paths(collection, name, [g[0] for g in group])
            pts = [p for g in group for p in g[3]["points"]]
            batches = [pts[k:k + EMBED_BATCH] for k in range(0, len(pts), EMBED_BATCH)]

            def work(batch):
                vecs = embed_with_retry(embedder, [p["embed_text"] for p in batch])
                q.upsert(collection, [{"id": p["id"], "vector": v, "payload": p["payload"]} for p, v in zip(batch, vecs)])
                return len(batch)

            with concurrent.futures.ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
                for fut in [pool.submit(work, b) for b in batches]:
                    fut.result()
            for rel, h, size, b in group:
                new_files[rel] = {"h": h, "size": size, "cv": b["cv"], "mv": mask_fp, "branch": branch,
                                  "n": len(b["points"]), "sprog": b["sprog"], "types": dict(b["types"])}
            save(complete=False)
    except Exception as exc:  # noqa: BLE001 - keep other repos alive, state is consistent up to the last group
        errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")

    rep.update(chunks=n_chunks, element_type=dict(types), sprog=dict(sprogs), masked_chunks=masked_n,
               points_deleted=n_deleted_pts, errors=errors)
    rep["commit_sha"] = commit
    if not args.dry_run:
        save(complete=(not capped and not errors))
    rep["status"] = "error" if errors else ("capped" if capped else "ok")
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def cmd_sync(args) -> int:
    collection = args.collection
    entries = load_registry(args.registry)
    if args.repo:
        names = set(args.repo)
        missing = names - {e["name"] for e in entries}
        if missing:
            print(json.dumps({"kanarie": KANARIE, "error": f"ukendt repo i registret: {sorted(missing)}"}))
            return 2
        entries = [e for e in entries if e["name"] in names]
    sd = state_dir(args, collection)
    mask_fp = mask_fingerprint()
    log_line(cmd="sync", event="start", collection=collection, repos=len(entries), dry_run=int(args.dry_run),
             max_files=args.max_files or 0, full=int(args.full), mask_fp=mask_fp)
    q = None if args.dry_run else Qdrant(args.qdrant)
    embedder = None if args.dry_run else make_embedder()
    results = []
    rc = 0
    lock = None if args.dry_run else RunLock(sd)
    try:
        if lock:
            lock.__enter__()
        for e in entries:
            try:
                rep = run_repo(e, args, collection, q, embedder, sd, mask_fp)
            except Exception as exc:  # noqa: BLE001
                rep = {"repo": e["name"], "kilde": kilde_of(e), "status": "error", "errors": [f"{type(exc).__name__}: {str(exc)[:200]}"]}
            if rep["status"] == "error":
                rc = 1
            results.append(rep)
            log_line(cmd="sync", repo=rep["repo"], kilde=rep.get("kilde"), status=rep["status"], dry_run=int(args.dry_run),
                     files=rep.get("files_to_process", 0), deleted_files=rep.get("files_deleted", 0), chunks=rep.get("chunks", 0),
                     sha=(rep.get("commit_sha") or "")[:10], reason=rep.get("reason", ""), errors=";".join(rep.get("errors", [])))
    finally:
        if lock:
            lock.__exit__(None, None, None)
    totals = collections.Counter()
    types: collections.Counter = collections.Counter()
    for r in results:
        totals["files_allowed"] += r.get("files_allowed", 0)
        totals["files_to_process"] += r.get("files_to_process", 0)
        totals["files_deleted"] += r.get("files_deleted", 0)
        totals["chunks"] += r.get("chunks", 0)
        types.update(r.get("element_type", {}))
    summary = {"kanarie": KANARIE, "cmd": "sync", "collection": collection, "dry_run": bool(args.dry_run), "ts": ts_iso(),
               "totals": dict(totals), "element_type": dict(types), "repos": results}
    if not args.dry_run:
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "_last_run.json").write_text(json.dumps({k: summary[k] for k in ("ts", "totals", "collection")} | {
            "repos": len(results), "errors": sum(1 for r in results if r["status"] == "error")}, indent=1), encoding="utf-8")
    log_line(cmd="sync", event="end", collection=collection, dry_run=int(args.dry_run), rc=rc, **{k: v for k, v in totals.items()})
    print(json.dumps(summary, indent=1))
    return rc


# --------------------------------------------------------------------------- status
def cmd_status(args) -> int:
    collection = args.collection
    sd = state_dir(args, collection)
    st_repos = []
    if sd.is_dir():
        for p in sorted(sd.glob("*.json")):
            if p.name.startswith("_"):
                continue
            try:
                st_repos.append(json.loads(p.read_text(encoding="utf-8")))
            except ValueError:
                continue
    last_run = {}
    try:
        last_run = json.loads((sd / "_last_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    out = {"kanarie": KANARIE, "cmd": "status", "ts": ts_iso(), "collection": collection,
           "qdrant": {"url": args.qdrant, "reachable": None, "points_count": None, "error": ""},
           "last_run": last_run, "per_kilde": {}, "per_element_type": {}, "per_sprog": {}, "per_repo": []}
    by_repo: dict[str, int] = {}
    if not args.no_qdrant:
        q = Qdrant(args.qdrant)
        try:
            info = q.collection(collection)
            out["qdrant"]["reachable"] = True
            if info is not None:
                out["qdrant"]["points_count"] = info.get("points_count")
                out["per_kilde"] = q.facet(collection, "kilde")
                out["per_element_type"] = q.facet(collection, "element_type")
                out["per_sprog"] = q.facet(collection, "sprog")
                by_repo = q.facet(collection, "repo")
            else:
                out["qdrant"]["error"] = "collection findes ikke"
        except Exception as exc:  # noqa: BLE001
            out["qdrant"].update(reachable=False, error=f"{type(exc).__name__}: {str(exc)[:160]}")
    for s in st_repos:
        out["per_repo"].append({
            "repo": s.get("repo"), "kilde": s.get("kilde"), "branch": s.get("branch"),
            "commit_sha": s.get("commit_sha"), "pending_sha": s.get("pending_sha", ""), "complete": s.get("complete"),
            "last_run_ts": s.get("last_run_ts"), "last_run_iso": s.get("last_run_iso"),
            "files": s.get("totals", {}).get("files"), "chunks_state": s.get("totals", {}).get("chunks"),
            "points": by_repo.get(s.get("repo")),
        })
    out["repos_in_state"] = len(st_repos)
    print(json.dumps(out, indent=1))
    log_line(cmd="status", collection=collection, points=out["qdrant"]["points_count"], repos=len(st_repos))
    return 0


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gigakode.py", description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--collection", default=os.environ.get("GIGAKODE_COLLECTION", DEFAULT_COLLECTION))
    common.add_argument("--qdrant", default=os.environ.get("GIGAKODE_QDRANT", DEFAULT_QDRANT))
    common.add_argument("--registry", default=os.environ.get("GIGAKODE_REGISTRY"))
    common.add_argument("--state-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", parents=[common])
    s.add_argument("--repo", action="append", help="only this registry name (repeatable)")
    s.add_argument("--dry-run", action="store_true", help="no Qdrant write, no embedding, no state write")
    s.add_argument("--max-files", type=int, default=0, help="process at most N changed files per repo (rest next run)")
    s.add_argument("--full", action="store_true", help="re-chunk + re-insert every file (ignores hashes)")
    st = sub.add_parser("status", parents=[common])
    st.add_argument("--no-qdrant", action="store_true")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    assert_hus_qdrant(args.qdrant)                  # also for --dry-run / --no-qdrant: a wrong URL is a config error
    if args.cmd == "sync":
        return cmd_sync(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
