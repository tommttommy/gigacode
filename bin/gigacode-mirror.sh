#!/usr/bin/env bash
# gigacode-mirror.sh — wrapper om bin/gigacode-mirror.py (spejl code__gigacode__main :16343 -> :16333).
# KANARIE=GIGACODE-INGEST-20261105
#
# STAGED: dette script er kun til sandkassen indtil Tommy giver GO til deploy + cron (deploy/gigacode-mirror.cron).
# Fail-closed: lock, guard-stier og python findes FOER nogen request sendes.
# Lock på lokal disk (/tmp), aldrig NFS. En overlappende kørsel forlader tavst med 0.
set -uo pipefail

KANARIE="GIGACODE-INGEST-20261105"
BASE="${GIGACODE_MIRROR_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_SCRIPT="$BASE/bin/gigacode-mirror.py"
LOCKDIR="${GIGACODE_LOCKDIR:-/tmp/gigakode-locks}"
COLLECTION="${GIGAKODE_MIRROR_COLLECTION:-code__gigacode__main}"
SOURCE="${GIGACODE_MIRROR_SOURCE:-http://127.0.0.1:16343}"
TARGET="${GIGACODE_MIRROR_TARGET:-http://127.0.0.1:16333}"
STATE_DIR="${GIGAKODE_STATE:-/tmp/opencode/gigakode-state}"
PY="${GIGACODE_PY:-python3}"

say() { echo "[gigacode-mirror] $(date -Iseconds) KANARIE=$KANARIE $*"; }

if [ ! -f "$PY_SCRIPT" ]; then
  say "FEJL: mangler $PY_SCRIPT — afbryder (fail-closed)"
  exit 1
fi

mkdir -p "$LOCKDIR" 2>/dev/null || { say "FEJL: kan ikke oprette $LOCKDIR — afbryder"; exit 1; }
exec 9>"$LOCKDIR/gigakode-mirror.lock"
flock -n 9 || exit 0

say "start collection=$COLLECTION source=$SOURCE target=$TARGET state=$STATE_DIR args=$*"
"$PY" -B "$PY_SCRIPT" --collection "$COLLECTION" --source "$SOURCE" --target "$TARGET" --state-dir "$STATE_DIR" "$@"
rc=$?
if [ "$rc" -eq 0 ]; then
  say "OK rc=0"
else
  say "FEJL rc=$rc"
fi
exit "$rc"
