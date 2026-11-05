"""Tests for gigakode.py (KANARIE=GIGAKODE-20261103).

Run:  python3 -m pytest <this file> -q --basetemp=<scratch dir>
Everything writes ONLY to the test collection `gigakode_test` on hus-qdrant (127.0.0.1:16343), recreated per
test and deleted afterwards. State/log/registry live in pytest tmp dirs. Embeddings are a deterministic fake
(GIGAKODE_EMBEDDER=fake) except test_real_embedding_smoke (LiteLLM embed-qwen3, 3 tiny chunks).
Synthetic values only - no real keys, no real CPR, no real names.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parent.parent / "bin"
sys.dont_write_bytecode = True
sys.path.insert(0, str(BIN))
import gigakode as g  # noqa: E402

QURL = "http://127.0.0.1:16343"
COLL = "gigakode_test"


# ----------------------------------------------------------------------------- helpers
def qreq(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(QURL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def drop_collection():
    assert COLL == "gigakode_test"          # never touch anything else
    try:
        qreq("DELETE", f"/collections/{COLL}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise


def scroll(flt=None):
    out, offset = [], None
    while True:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if flt:
            body["filter"] = flt
        if offset is not None:
            body["offset"] = offset
        res = qreq("POST", f"/collections/{COLL}/points/scroll", body)["result"]
        out += res["points"]
        offset = res.get("next_page_offset")
        if offset is None:
            return out


def count(flt=None):
    body = {"exact": True}
    if flt:
        body["filter"] = flt
    return qreq("POST", f"/collections/{COLL}/points/count", body)["result"]["count"]


def must(**kv):
    return {"must": [{"key": k, "match": {"value": v}} for k, v in kv.items()]}


def sh(*cmd, cwd):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@test.invalid", "-c", "user.name=t", *cmd],
                   check=True, capture_output=True)


def commit_all(repo: Path, msg="c"):
    sh("add", "-A", cwd=repo)
    sh("commit", "-q", "-m", msg, cwd=repo)


def write(repo: Path, rel: str, text: str):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


PY_MAIN = '''import os

def alpha(x):
    return x + 1


class Beta:
    def gamma(self):
        return 2

    def delta(self):
        return 3


@app.get("/health")
async def health():
    return {"ok": True}
'''


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGAKODE_EMBEDDER", "fake")
    monkeypatch.setenv("GIGAKODE_B1_STATE", str(tmp_path / "b1state"))
    monkeypatch.setattr(g, "LOG_PATH", tmp_path / "gigakode.log")
    drop_collection()
    repos = tmp_path / "repos"
    repos.mkdir()
    reg = tmp_path / "registry.json"
    state = tmp_path / "state"

    class E:
        pass

    e = E()
    e.tmp, e.repos, e.reg, e.state = tmp_path, repos, reg, state
    e.entries = []

    def add_repo(name, files: dict, github=True, kilde_entry=None, git=True, **extra):
        root = repos / name
        root.mkdir()
        for rel, txt in files.items():
            write(root, rel, txt)
        ent = {"name": name, "root": str(root), "branch": "main", "enabled": True, "b1_target": "hus"}
        if git:
            sh("init", "-q", "-b", "main", cwd=root)
            commit_all(root, "init")
            ent.update(git_root=str(root), branch_from_git=True)
        else:
            ent.update(git_root=None, branch_from_git=False)
        if github:
            ent["github"] = f"x/{name}"
        ent.update(extra)
        e.entries.append(ent)
        reg.write_text(json.dumps({"repos": e.entries}), encoding="utf-8")
        return root

    def sync(*extra, capsys=None):
        argv = ["sync", "--collection", COLL, "--qdrant", QURL, "--registry", str(reg), "--state-dir", str(state), *extra]
        rc = g.main(argv)
        return rc

    def status(*extra):
        return g.main(["status", "--collection", COLL, "--qdrant", QURL, "--registry", str(reg), "--state-dir", str(state), *extra])

    e.add_repo, e.sync, e.status = add_repo, sync, status
    yield e
    drop_collection()


def run_json(capsys, fn, *a):
    capsys.readouterr()
    rc = fn(*a)
    out = capsys.readouterr().out
    return rc, json.loads(out)


# ----------------------------------------------------------------------------- 1. chunk -> payload fields
def test_chunk_payload_fields(env, capsys):
    root = env.add_repo("t__fields", {"app/main.py": PY_MAIN, "README.md": "# Titel\n\ntekst her\n"})
    rc, out = run_json(capsys, env.sync)
    assert rc == 0 and out["kanarie"] == "GIGAKODE-20261103"
    pts = scroll(must(repo="t__fields"))
    assert len(pts) >= 5
    required = {"repo", "branch", "commit_sha", "path", "line_start", "line_end", "element_type", "symbol", "sprog",
                "kilde", "text", "chunker_version"}
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for p in pts:
        pl = p["payload"]
        assert required <= set(pl), required - set(pl)
        assert pl["repo"] == "t__fields" and pl["branch"] == "main" and pl["kilde"] == "hus"
        assert pl["commit_sha"] == head
        assert isinstance(pl["line_start"], int) and pl["line_end"] >= pl["line_start"]
        assert pl["element_type"] in g.ELEMENT_TYPES
        assert pl["chunker_version"] >= 1 and pl["text"].strip()
    by_sym = {p["payload"]["symbol"]: p["payload"] for p in pts if p["payload"]["path"] == "app/main.py"}
    assert by_sym["alpha"]["line_start"] == 3 and by_sym["alpha"]["sprog"] == "python"
    assert by_sym["alpha"]["signature"].startswith("def alpha")
    assert by_sym["health"]["route"].startswith('@app.get("/health")')
    # filters on the payload indexes work (isolation = filter, like patient_lnr)
    assert count(must(repo="t__fields", element_type="class")) == 1
    assert count({"must": [{"key": "line_start", "range": {"gte": 3, "lte": 3}}]}) >= 1
    # collection config: 2560d cosine, vectors on disk, payload indexes present
    info = qreq("GET", f"/collections/{COLL}")["result"]
    vec = info["config"]["params"]["vectors"]
    assert vec["size"] == 2560 and vec["distance"] == "Cosine" and vec["on_disk"] is True
    schema = info["payload_schema"]
    for f in g.KEYWORD_FIELDS:
        assert schema[f]["data_type"] == "keyword", f
    assert schema["line_start"]["data_type"] == "integer"


# ----------------------------------------------------------------------------- 2. idempotency
def test_idempotent_two_runs_same_count(env, capsys):
    env.add_repo("t__idem", {"a.py": PY_MAIN, "b.py": "def only():\n    return 1\n", "c.md": "# H1\n\nx\n\n## H2\n\ny\n"})
    rc, o1 = run_json(capsys, env.sync)
    n1 = count()
    ids1 = sorted(str(p["id"]) for p in scroll())
    rc, o2 = run_json(capsys, env.sync)
    assert count() == n1 and n1 > 0
    assert o2["repos"][0]["status"] == "unchanged"          # sha gate: nothing re-embedded
    # forced full re-insert (delete + upsert of the same deterministic ids) is still the same set
    rc, o3 = run_json(capsys, env.sync, "--full")
    assert o3["repos"][0]["status"] == "ok" and o3["repos"][0]["files_to_process"] == 3
    assert count() == n1
    assert sorted(str(p["id"]) for p in scroll()) == ids1


def test_idempotent_non_git_and_local_repo(env, capsys):
    env.add_repo("t__nogit", {"x.py": "def f():\n    return 1\n"}, github=False, git=False)
    run_json(capsys, env.sync)
    n1 = count()
    _, o2 = run_json(capsys, env.sync)
    assert count() == n1 and o2["repos"][0]["files_to_process"] == 0
    st = g.load_state(g.state_dir(type("A", (), {"state_dir": str(env.state)})(), COLL), "t__nogit")
    assert st["commit_sha"].startswith("nogit-") and st["complete"] is True


# ----------------------------------------------------------------------------- 3. incremental delete of changed file
def test_incremental_replaces_changed_file_and_deletes_removed(env, capsys):
    root = env.add_repo("t__inc", {"a.py": PY_MAIN, "keep.py": "def keeper():\n    return 1\n", "gone.py": "def gone_fn():\n    return 9\n"})
    run_json(capsys, env.sync)
    keep_before = {str(p["id"]): p["payload"]["indekseret_ts"] for p in scroll(must(repo="t__inc", path="keep.py"))}
    assert keep_before and count(must(repo="t__inc", path="gone.py")) == 1
    syms_before = {p["payload"]["symbol"] for p in scroll(must(repo="t__inc", path="a.py"))}
    assert {"alpha", "gamma", "delta"} <= syms_before
    sha_before = g.load_state(env.state / COLL, "t__inc")["commit_sha"]
    # change a.py: drop `alpha`, shift lines, add `omega`; delete gone.py; add new.py
    write(root, "a.py", "def omega():\n    return 0\n\n\nclass Beta:\n    def gamma(self):\n        return 2\n")
    (root / "gone.py").unlink()
    write(root, "new.py", "def fresh():\n    return 5\n")
    commit_all(root, "change")
    rc, out = run_json(capsys, env.sync)
    rep = out["repos"][0]
    assert rep["mode"] == "diff" and rep["status"] == "ok"        # git-diff-narrowed path
    assert rep["files_to_process"] == 2 and rep["files_deleted"] == 1
    syms_after = {p["payload"]["symbol"] for p in scroll(must(repo="t__inc", path="a.py"))}
    assert "omega" in syms_after and "alpha" not in syms_after and "delta" not in syms_after
    assert count(must(repo="t__inc", path="gone.py")) == 0
    assert count(must(repo="t__inc", path="new.py")) == 1
    keep_after = {str(p["id"]): p["payload"]["indekseret_ts"] for p in scroll(must(repo="t__inc", path="keep.py"))}
    assert keep_after == keep_before                                # untouched file was not re-embedded
    st = g.load_state(env.state / COLL, "t__inc")
    assert st["commit_sha"] != sha_before and "gone.py" not in st["files"]
    # no point of the repo carries an outdated commit_sha for a changed path
    assert {p["payload"]["commit_sha"] for p in scroll(must(repo="t__inc", path="a.py"))} == {st["commit_sha"]}
    # and a second run is idempotent again
    n = count()
    run_json(capsys, env.sync)
    assert count() == n


def test_scan_path_when_tree_dirty_local_repo(env, capsys):
    root = env.add_repo("t__local", {"a.py": "def one():\n    return 1\n"}, github=False)
    run_json(capsys, env.sync)
    write(root, "a.py", "def two():\n    return 2\n")             # uncommitted edit on a local (non-mirror) repo
    _, out = run_json(capsys, env.sync)
    assert out["repos"][0]["mode"] == "scan" and out["repos"][0]["files_to_process"] == 1
    assert {p["payload"]["symbol"] for p in scroll(must(repo="t__local"))} == {"two"}


def test_max_files_cap_then_resume(env, capsys):
    env.add_repo("t__cap", {f"f{i}.py": f"def fn{i}():\n    return {i}\n" for i in range(5)})
    _, o1 = run_json(capsys, env.sync, "--max-files", "2")
    r1 = o1["repos"][0]
    assert r1["status"] == "capped" and r1["files_to_process"] == 2 and count() == 2
    st = g.load_state(env.state / COLL, "t__cap")
    assert st["complete"] is False and st["commit_sha"] == "" and st["pending_sha"]
    _, o2 = run_json(capsys, env.sync, "--max-files", "10")
    assert o2["repos"][0]["files_to_process"] == 3 and count() == 5
    assert g.load_state(env.state / COLL, "t__cap")["complete"] is True


# ----------------------------------------------------------------------------- 4. masking
def test_masking_before_payload_and_embed(env, capsys, monkeypatch):
    # all values synthetic
    fake_key = "sk-" + "ZETATEST" * 4
    fake_ghp = "ghp_" + "Q1w2E3r4T5y6U7i8O9p0" + "AaBbCcDdEeFf"
    fake_cpr = "010199-9999"
    fake_mail = "zeta.testperson@fake-klinik.dk"
    fake_pw = "password = 'Zx9Yw8Vu7Ts6Rq5'"
    src = (
        "def configure():\n"
        f"    api_key = '{fake_key}'\n"
        f"    gh = '{fake_ghp}'\n"
        f"    patient = '{fake_cpr}'\n"
        f"    owner = '{fake_mail}'\n"
        f"    {fake_pw}\n"
        "    return api_key\n"
    )
    env.add_repo("t__mask", {"cfg.py": src})
    seen_texts: list[str] = []

    class Rec(g.FakeEmbedder):
        def embed_batch(self, texts):
            seen_texts.extend(texts)
            return super().embed_batch(texts)

    monkeypatch.setattr(g, "make_embedder", lambda: Rec())
    rc, _ = run_json(capsys, env.sync)
    assert rc == 0
    pts = scroll(must(repo="t__mask"))
    assert pts and seen_texts
    secrets = [fake_key, fake_ghp, fake_cpr.replace("-", ""), fake_cpr, fake_mail, "Zx9Yw8Vu7Ts6Rq5", "ZETATEST" * 4]
    blob_payload = json.dumps([p["payload"] for p in pts])
    blob_embed = "\n".join(seen_texts)
    for s in secrets:
        assert s not in blob_payload, f"umaskeret i payload: {s[:6]}..."
        assert s not in blob_embed, f"umaskeret i embed-tekst: {s[:6]}..."
    assert any("MASKED" in p["payload"]["text"] for p in pts)
    assert any(p["payload"]["sha256"] for p in pts)


# ----------------------------------------------------------------------------- 5. element_type classification
JS_SRC = '''const express = require('express');
const app = express();

function helper(a) {
  return a;
}

class Widget {
  render() {
    return 1;
  }
}

app.get('/ping', (req, res) => {
  res.send('pong');
});
'''


def build(env, name, files, **extra):
    root = env.add_repo(name, files, **extra)
    entry = env.entries[-1]
    out = {}
    for rel in files:
        b = g.build_points(entry, root, rel, "hus", "main", "deadbeef")
        out[rel] = [(pt["payload"]["symbol"], pt["payload"]["element_type"], pt["payload"]["line_start"]) for pt in b["points"]]
    return out


def test_element_type_classification(env):
    files = {
        "pkg/mod.py": PY_MAIN,
        "tests/test_mod.py": "def test_x():\n    assert 1\n",
        "pkg/widget.test.ts": "function t() {\n  return 1;\n}\n",
        "doc.md": "# Overskrift\n\ntekst\n\n## Under\n\nmere\n",
        "conf.yaml": "services:\n  web:\n    image: x\nvolumes:\n  data: {}\n",
        "run.sh": "#!/bin/sh\nstart() {\n  echo hi\n}\n",
        "web.js": JS_SRC,
        "script.py": "x = 1\nprint(x)\n",
    }
    got = build(env, "t__types", files)
    py = {s: t for s, t, _ in got["pkg/mod.py"]}
    assert py["alpha"] == "function" and py["Beta"] == "class" and py["gamma"] == "method" and py["health"] == "route"
    assert {t for _, t, _ in got["tests/test_mod.py"]} == {"test"}
    assert {t for _, t, _ in got["pkg/widget.test.ts"]} == {"test"}
    assert {t for _, t, _ in got["doc.md"]} == {"doc"}
    assert {t for _, t, _ in got["conf.yaml"]} == {"config"}
    assert {t for _, t, _ in got["run.sh"]} == {"function"}
    js = {s: t for s, t, _ in got["web.js"]}
    assert js["helper"] == "function" and js["Widget"] == "class"
    assert any(t == "route" for s, t, _ in got["web.js"]), got["web.js"]
    assert {t for _, t, _ in got["script.py"]} == {"other"}                # no defs -> window chunk
    assert all(t in g.ELEMENT_TYPES for v in got.values() for _, t, _ in v)


def test_split_continuation_inherits_kind(env):
    body = "\n".join(f"    v{i} = {i}" for i in range(200))
    root = env.add_repo("t__split", {"big.py": f"def huge():\n{body}\n    return 1\n"})
    entry = env.entries[-1]
    b = g.build_points(entry, root, "big.py", "hus", "main", "x")
    assert len(b["points"]) >= 2
    assert {pt["payload"]["element_type"] for pt in b["points"]} == {"function"}
    assert {pt["payload"]["symbol"] for pt in b["points"]} == {"huge"}
    assert sorted(pt["payload"]["part"] for pt in b["points"])[:2] == [1, 2]
    assert len({pt["id"] for pt in b["points"]}) == len(b["points"])      # distinct deterministic ids


def test_same_start_line_chunks_get_distinct_ids(env, monkeypatch):
    root = env.add_repo("t__coll", {"R.md": "\n".join(f"line {i}" for i in range(10)) + "\n"})
    entry = env.entries[-1]
    C = g.cr.Chunk
    monkeypatch.setattr(g.cr, "chunk_file", lambda p: [C("big", 1, 10, "\n".join(f"line {i}" for i in range(10))),
                                                      C("big#2", 5, 10, "x5"), C("sec", 5, 8, "y5")])
    b1 = g.build_points(entry, root, "R.md", "hus", "main", "x")
    b2 = g.build_points(entry, root, "R.md", "hus", "main", "x")
    ids = [p["id"] for p in b1["points"]]
    assert len(ids) == 3 and len(set(ids)) == 3                  # no silent overwrite
    assert ids == [p["id"] for p in b2["points"]]                # still deterministic
    assert ids[1] == g.point_id("t__coll", "main", "R.md", 5, g.cr.chunker_version("markdown"))   # first keeps the plain formula


def test_deterministic_point_id():
    a = g.point_id("r", "main", "p.py", 10, 2)
    assert a == g.point_id("r", "main", "p.py", 10, 2)
    assert a != g.point_id("r", "main", "p.py", 11, 2) and a != g.point_id("r", "main", "p.py", 10, 3)
    assert a != g.point_id("r", "dev", "p.py", 10, 2) and a != g.point_id("r2", "main", "p.py", 10, 2)


# ----------------------------------------------------------------------------- 6. registry, kilde, gates, dry-run, status
def test_kilde_mapping():
    assert g.kilde_of({"name": "hus__ekstern__a__b", "kind": "ekstern", "b1_target": "ekstern"}) == "ekstern"
    assert g.kilde_of({"name": "hus__altona", "b1_target": "hus"}) == "hus"
    assert g.kilde_of({"name": "hus__hippocampus", "b1_target": "hus"}) == "hippocampus"
    assert g.kilde_of({"name": "voice-web", "b1_target": "director"}) == "projekt"
    assert g.kilde_of({"name": "x", "gigakode_kilde": "hippocampus"}) == "hippocampus"


def test_new_repo_in_registry_picked_up_next_run(env, capsys):
    env.add_repo("t__first", {"a.py": "def a():\n    return 1\n"})
    run_json(capsys, env.sync)
    assert count(must(repo="t__second")) == 0
    env.add_repo("t__second", {"b.py": "def b():\n    return 2\n"}, b1_target="ekstern", kind="ekstern")
    _, out = run_json(capsys, env.sync)
    names = {r["repo"]: r["status"] for r in out["repos"]}
    assert names == {"t__first": "unchanged", "t__second": "ok"}
    assert {p["payload"]["kilde"] for p in scroll(must(repo="t__second"))} == {"ekstern"}
    _, out = run_json(capsys, env.sync, "--repo", "t__first")
    assert [r["repo"] for r in out["repos"]] == ["t__first"]


def test_dry_run_writes_nothing(env, capsys):
    env.add_repo("t__dry", {"a.py": PY_MAIN, "README.md": "# t\n\nx\n"})
    rc, out = run_json(capsys, env.sync, "--dry-run")
    rep = out["repos"][0]
    assert rc == 0 and out["dry_run"] is True
    assert rep["files_to_process"] == 2 and rep["chunks"] > 0 and sum(rep["element_type"].values()) == rep["chunks"]
    assert not env.state.exists() or not any(env.state.rglob("*.json"))
    with pytest.raises(urllib.error.HTTPError) as ei:               # collection was never created
        qreq("GET", f"/collections/{COLL}")
    assert ei.value.code == 404


def test_disabled_and_missing_root_and_b1_block(env, capsys):
    env.add_repo("t__off", {"a.py": "def a():\n    return 1\n"}, enabled=False)
    env.add_repo("t__noroot", {"a.py": "def a():\n    return 1\n"})
    env.entries[-1]["root"] = str(env.repos / "does-not-exist")
    env.add_repo("t__blocked", {"a.py": "def a():\n    return 1\n"})
    (env.tmp / "b1state").mkdir()
    (env.tmp / "b1state" / "t__blocked.sync.json").write_text(json.dumps({"scan": "RED", "ok": 0}))
    _, out = run_json(capsys, env.sync)
    st = {r["repo"]: (r["status"], r.get("reason", "")) for r in out["repos"]}
    assert st["t__off"][0] == "skipped" and st["t__noroot"][0] == "skipped"
    assert st["t__blocked"] == ("blocked", "b1-scan RED")
    with pytest.raises(urllib.error.HTTPError) as ei:               # nothing passed the gates -> no collection, no points
        count()
    assert ei.value.code == 404


def test_status_json_and_kanarie(env, capsys):
    env.add_repo("t__st1", {"a.py": PY_MAIN})
    env.add_repo("t__st2", {"b.py": "def b():\n    return 1\n"}, b1_target="ekstern", kind="ekstern")
    run_json(capsys, env.sync)
    rc, st = run_json(capsys, env.status)
    assert rc == 0 and st["kanarie"] == "GIGAKODE-20261103" and st["collection"] == COLL
    assert st["qdrant"]["reachable"] is True and st["qdrant"]["points_count"] == count()
    assert set(st["per_kilde"]) == {"hus", "ekstern"} and sum(st["per_kilde"].values()) == count()
    repos = {r["repo"]: r for r in st["per_repo"]}
    assert set(repos) == {"t__st1", "t__st2"}
    r1 = repos["t__st1"]
    assert r1["points"] == count(must(repo="t__st1")) and r1["commit_sha"] and r1["complete"] is True
    assert r1["last_run_ts"] and r1["kilde"] == "hus"
    assert st["last_run"]["ts"]
    assert sum(st["per_element_type"].values()) == count()
    # --no-qdrant still gives state-only JSON
    _, st2 = run_json(capsys, env.status, "--no-qdrant")
    assert st2["qdrant"]["reachable"] is None and len(st2["per_repo"]) == 2


def test_kanarie_in_every_log_line(env, capsys):
    env.add_repo("t__log", {"a.py": "def a():\n    return 1\n"})
    run_json(capsys, env.sync)
    run_json(capsys, env.sync, "--dry-run")
    run_json(capsys, env.status)
    lines = (env.tmp / "gigakode.log").read_text().splitlines()
    assert len(lines) >= 6 and all("KANARIE=GIGAKODE-20261103" in ln for ln in lines)


def test_mask_rule_change_triggers_rechunk(env, capsys, monkeypatch):
    env.add_repo("t__mfp", {"a.py": "def a():\n    return 1\n"})
    run_json(capsys, env.sync)
    _, o = run_json(capsys, env.sync)
    assert o["repos"][0]["status"] == "unchanged"
    monkeypatch.setattr(g, "mask_fingerprint", lambda: "changedrules1")
    _, o = run_json(capsys, env.sync)
    assert o["repos"][0]["status"] == "ok" and o["repos"][0]["files_to_process"] == 1


# ----------------------------------------------------------------------------- 7. real embedding (LiteLLM embed-qwen3)
def test_real_embedding_smoke(env, capsys, monkeypatch):
    monkeypatch.delenv("GIGAKODE_EMBEDDER", raising=False)
    env.add_repo("t__real", {
        "a.py": "def parse_cpr_checksum(value):\n    return sum(map(int, value))\n\n\ndef open_socket(host):\n    return host\n",
    })
    rc, out = run_json(capsys, env.sync)
    assert rc == 0 and out["repos"][0]["chunks"] == 2
    emb = g.make_embedder()
    vec = emb.embed_batch(["function that sums the digits of a value"])[0]
    assert len(vec) == 2560
    hits = qreq("POST", f"/collections/{COLL}/points/search",
                {"vector": vec, "limit": 1, "with_payload": True, "filter": must(repo="t__real")})["result"]
    assert hits and hits[0]["payload"]["symbol"] == "parse_cpr_checksum"


# ----------------------------------------------------------------------------- 8. Helios ban (Tommy 2026-11-03 22:44 UTC)
@pytest.mark.parametrize("url", ["http://127.0.0.1:6333", "http://127.0.0.1:6340", "http://localhost:6337",
                                 "http://127.0.0.1:8700", "http://qdrant-helios:6333", "http://10.12.38.142:16343",
                                 "http://127.0.0.1:16333"])
def test_helios_and_foreign_qdrant_refused(url, capsys):
    with pytest.raises(SystemExit):
        g.main(["status", "--qdrant", url, "--no-qdrant"])
    with pytest.raises(SystemExit):
        g.main(["sync", "--dry-run", "--qdrant", url])
    with pytest.raises(SystemExit):
        g.Qdrant(url)


def test_only_hus_qdrant_url_in_source():
    src = (BIN / "gigakode.py").read_text()
    import re
    urls = set(re.findall(r"https?://[\w.\-]+:(\d+)", src))
    assert urls <= {"16343"}, urls                  # the only literal host:port in the code is :16343
    assert g.DEFAULT_QDRANT == "http://127.0.0.1:16343"
    assert g.assert_hus_qdrant("http://127.0.0.1:16343/") == "http://127.0.0.1:16343"
