"""Tests for GIGACODE-INGEST-20261105 (registry-post for gigacode + sync-wrapper + staged cron).

KANARIE=GIGACODE-INGEST-20261105

Run (fra /shared/claude-shared/director-v3/sandbox/gigacode-ingest-20261105):
  python3 -B -m pytest tests/test_gigacode_ingest.py -v -p no:cacheprovider

Alt er offline: INGEN netkald, INGEN Qdrant-skrivning, INGEN cron-installation. Registry, wrapper og
cron laeses som filer; gigakode.py importeres READ ONLY fra gigakode-20261103-sandkassen.
Syntetiske vaerdier only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
REG_GIGACODE = BASE / "etc" / "repos-gigacode.json"
REG_B1 = Path("/shared/claude-shared/code-rag/etc/repos-b1.json")
SYNC = BASE / "bin" / "gigacode-sync.sh"
CRON = BASE / "deploy" / "gigacode-sync.cron"
GIGAKODE_BIN = BASE.parent / "gigakode-20261103" / "bin"

sys.dont_write_bytecode = True
sys.path.insert(0, str(GIGAKODE_BIN))
import gigakode as g  # noqa: E402  (READ ONLY)

REQUIRED_KEYS = ("name", "root", "git_root", "branch", "branch_from_git", "enabled", "test",
                 "exclude", "qdrant_url", "b1_target", "gigakode_kilde")


def load_gigacode_registry() -> dict:
    return json.loads(REG_GIGACODE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 1
def test_registry_is_valid_json_with_exactly_one_post():
    data = load_gigacode_registry()
    for top in ("version", "comment", "qdrant_url", "embed_url", "embed_model", "vector_size", "b1", "repos"):
        assert top in data, f"top-noegle mangler: {top}"
    assert data["vector_size"] == 2560, "vector size er frossen (embed-qwen3)"
    assert data["embed_model"] == "embed-qwen3"
    assert len(data["repos"]) == 1, f"forventede praecis 1 post, fik {len(data['repos'])}"
    post = data["repos"][0]
    missing = [k for k in REQUIRED_KEYS if k not in post]
    assert not missing, f"noegler mangler i posten: {missing}"
    assert post["name"] == "gigacode"
    assert post["branch"] == "main"
    assert post["branch_from_git"] is True
    assert post["enabled"] is True
    assert post["test"] is False
    assert post["exclude"] == []
    # ingen github-noegle: der findes intet GitHub-repo tommttommy/gigacode (verificeret via API)
    assert "github" not in post, "lokal post maa ikke have github-noegle"
    # klonrod er en dokumenteret placeholder
    assert post["root"] == "/home/tlg/code-rag-clones/gigacode"
    assert post["git_root"] == post["root"]
    assert "placeholder" in data["comment"].lower() or "PLACEHOLDER" in data["comment"]


# --------------------------------------------------------------------------- 2
def test_load_registry_merges_b1_and_gigacode():
    entries = g.load_registry(f"{REG_B1},{REG_GIGACODE}")
    names = [e["name"] for e in entries]
    assert "gigacode" in names, "load_registry pickede ikke gigacode op"
    assert len(names) == len(set(names)), "load_registry gav dubletter"
    assert len(entries) >= 60, f"forventede >=60 poster (59 B1 + gigacode), fik {len(entries)}"
    post = next(e for e in entries if e["name"] == "gigacode")
    assert post["qdrant_url"] == "http://127.0.0.1:16343"


# --------------------------------------------------------------------------- 3
def test_kilde_of_is_hus():
    post = load_gigacode_registry()["repos"][0]
    assert g.kilde_of(post) == "hus"


# --------------------------------------------------------------------------- 4
def test_qdrant_url_is_16343_and_accepted():
    post = load_gigacode_registry()["repos"][0]
    assert post["qdrant_url"] == "http://127.0.0.1:16343"
    assert g.assert_hus_qdrant(post["qdrant_url"]) == "http://127.0.0.1:16343"
    assert 16343 not in g.FORBIDDEN_PORTS


# --------------------------------------------------------------------------- 5 (4 negative tests: begge sider af doeren)
@pytest.mark.parametrize("url", [
    "http://127.0.0.1:16333",   # near-miss port
    "http://127.0.0.1:6333",    # Helios/system Qdrant
    "http://127.0.0.1:6340",    # forbudt port
    "http://127.0.0.1:8700",    # forbudt port
])
def test_assert_hus_qdrant_refuses_foreign_urls(url):
    with pytest.raises(SystemExit) as exc:
        g.assert_hus_qdrant(url)
    assert "afvist" in str(exc.value)


# --------------------------------------------------------------------------- 6
def test_sync_wrapper_syntax_and_flock_line():
    r = subprocess.run(["bash", "-n", str(SYNC)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"bash -n fejlede: {r.stderr}"
    text = SYNC.read_text(encoding="utf-8")
    assert "set -uo pipefail" in text
    assert "exec 9>" in text
    assert "flock -n 9 || exit 0" in text
    assert "/tmp/gigakode-locks" in text, "flock skal ligge paa lokal disk, ikke NFS"
    assert "GIGACODE-INGEST-20261105" in text, "kanarie mangler"
    assert "--registry" in text and "--collection" in text
    assert "--force" in text and "--dry-run" in text


# --------------------------------------------------------------------------- 7
def test_sync_wrapper_fails_closed_on_missing_registry(tmp_path):
    env = dict(os.environ)
    env["GIGACODE_REGISTRY"] = str(tmp_path / "findes-ikke.json")
    env["GIGACODE_REG_B1"] = str(REG_B1)
    env["GIGACODE_LOCKDIR"] = str(tmp_path / "locks")
    env["GIGAKODE_STATE"] = str(tmp_path / "state")
    r = subprocess.run(["bash", str(SYNC)], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 1, f"forventede exit 1, fik {r.returncode} (stdout={r.stdout!r} stderr={r.stderr!r})"
    out = r.stdout + r.stderr
    assert "registry-fil mangler" in out, out
    assert "GIGACODE-INGEST-20261105" in out, "kanarie mangler i fejlbeskeden"


# --------------------------------------------------------------------------- 8
def test_cron_file_is_staged_with_exactly_one_active_line():
    text = CRON.read_text(encoding="utf-8")
    assert "STAGED" in text
    assert "Tommy-GO" in text
    assert "GIGACODE-INGEST-20261105" in text
    active = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert len(active) == 1, f"forventede praecis 1 aktiv linje, fik {len(active)}: {active}"
    assert active[0].startswith("*/5 * * * * "), active[0]
    assert str(SYNC) in active[0]
    assert ">> /tmp/gigakode-cron.log 2>&1" in active[0]


# --------------------------------------------------------------------------- 9
def test_entry_fingerprint_is_deterministic():
    post = load_gigacode_registry()["repos"][0]
    a = g.entry_fingerprint(post, "main")
    b = g.entry_fingerprint(json.loads(json.dumps(post)), "main")
    assert a == b, "entry_fingerprint er ikke deterministisk for samme post"
    assert len(a) == 12
    # negativ side: en aendret kilde skal give et andet fingerprint
    other = dict(post, gigakode_kilde="ekstern")
    assert g.entry_fingerprint(other, "main") != a
    assert g.entry_fingerprint(post, "dev") != a
