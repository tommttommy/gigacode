#!/usr/bin/env bash
# gigacode-sync.sh — cron/auto-sync wrapper for GIGAKODE. KANARIE=GIGACODE-INGEST-20261105
# Usage: gigacode-sync.sh [--dry-run] [--force] [--repo NAME] [--max-files N] [--full] [--status]
#
# * Registry = repos-b1.json (de 59 B1-poster) + repos-gigacode.json (den separate gigacode-post).
#   Begge sendes som ét kommasepareret --registry-led til gigakode.py (load_registry :85-96;
#   foerste definition af et navn vinder). Intet skrives til repos-b1.json.
# * Én flock per collection paa LOKAL disk (/tmp, ikke NFS): en overlappende cron-start
#   forlader med exit 0 uden at roere Qdrant. gigakode.py har desuden sin egen RunLock paa state-dir.
# * Fail-closed: mangler en registry-fil, eller er gigakode.py vaek, afbrydes der med exit 1.
# * --force mappes til gigakode.py's faktiske flag --full (tvang: re-chunk + re-indsaet alt).
# * Alt er env-styret; defaults er de eneste hardkodede stier (${VAR:-default}).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${GIGACODE_BASE:-$(cd "$HERE/.." && pwd)}"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1

CODE_RAG="${GIGACODE_CODE_RAG:-/shared/claude-shared/code-rag}"
REG_B1="${GIGACODE_REG_B1:-$CODE_RAG/etc/repos-b1.json}"
REG_GIGACODE="${GIGACODE_REGISTRY:-$BASE/etc/repos-gigacode.json}"
COLLECTION="${GIGAKODE_COLLECTION:-gigakode}"
QDRANT="${GIGAKODE_QDRANT:-http://127.0.0.1:16343}"
LOCKDIR="${GIGACODE_LOCKDIR:-/tmp/gigakode-locks}"
STATE_DIR="${GIGAKODE_STATE:-$BASE/state}"
GIGAKODE_PY="${GIGACODE_PY:-$(cd "$HERE/.." && pwd)/../gigakode-20261103/bin/gigakode.py}"
LOG_TAG="KANARIE=GIGACODE-INGEST-20261105"

say() { echo "[gigacode-sync] $(date -Iseconds) $LOG_TAG $*"; }

# ---- fail-closed prechecks (foer lock: en manglende fil er en konfigurationsfejl, ikke en kollision)
if [ ! -f "$REG_GIGACODE" ]; then
  say "FAIL: registry-fil mangler: $REG_GIGACODE — afbryder (exit 1)"
  exit 1
fi
if [ ! -f "$REG_B1" ]; then
  say "FAIL: registry-fil mangler: $REG_B1 — afbryder (exit 1)"
  exit 1
fi
if [ ! -f "$GIGAKODE_PY" ]; then
  say "FAIL: gigakode.py mangler: $GIGAKODE_PY — afbryder (exit 1)"
  exit 1
fi
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$REG_GIGACODE" || {
  say "FAIL: $REG_GIGACODE er ikke gyldig JSON — afbryder (exit 1)"; exit 1; }

# ---- argument-normalisering: --force -> --full, --status -> status-subcommand
CMD="sync"
ARGS=()
for a in "$@"; do
  case "$a" in
    --force)  ARGS+=("--full") ;;
    --status) CMD="status"; ARGS+=("--no-qdrant") ;;
    *)        ARGS+=("$a") ;;
  esac
done

mkdir -p "$LOCKDIR" 2>/dev/null || { say "FAIL: kan ikke oprette lockdir $LOCKDIR (exit 1)"; exit 1; }
mkdir -p "$STATE_DIR" 2>/dev/null || { say "FAIL: kan ikke oprette state-dir $STATE_DIR (exit 1)"; exit 1; }

# ---- flock per collection paa lokal disk
exec 9>"$LOCKDIR/gigacode-$COLLECTION.lock"
flock -n 9 || exit 0   # en anden koersel for samme collection ejer laasen: tavst exit 0 (som b1-gh-sync.sh)

REGISTRY="$REG_B1,$REG_GIGACODE"
say "start cmd=$CMD collection=$COLLECTION qdrant=$QDRANT registry=$REGISTRY args=${ARGS[*]:-none}"

python3 -B "$GIGAKODE_PY" "$CMD" --registry "$REGISTRY" --collection "$COLLECTION" --qdrant "$QDRANT" \
  --state-dir "$STATE_DIR" "${ARGS[@]:-}"
RC=$?
if [ "$RC" -eq 0 ]; then
  say "OK cmd=$CMD collection=$COLLECTION rc=0"
else
  say "FEJL cmd=$CMD collection=$COLLECTION rc=$RC"
fi
exit "$RC"
