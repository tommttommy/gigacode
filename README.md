# gigacode

Code-to-RAG indeksering for Hippocampus. Synkroniserer kode fra et repo ind i en Qdrant-collection (2560d, embed-qwen3).

KANARIE: GIGACODE-INGEST-20261105

## Struktur
- `bin/gigakode.py` — kerne: kode -> chunk -> embed -> upsert (collection + state + manifest).
- `bin/gigacode-sync.sh` — wrapper med flock; kører `gigakode.py sync` mod registry.
- `bin/gigacode-mirror.py` — spejl :16343 -> :16333 (hus -> hippo Qdrant), 1:1 punktkopi med gate-felter.
- `bin/gigacode-mirror.sh` — wrapper med flock for spejlet.
- `etc/repos-gigacode.json` — registry (én post: gigacode).
- `deploy/*.cron` — STAGED cron-filer (aktiveres kun med Tommys GO).
- `tests/` — offline tests (gigakode 27 + ingest 12 + mirror 25).

## Krav
- Mål-Qdrant: hippo :16333 (aldrig :6333/:6340 Helios).
- Embed: LiteLLM :4000 + embed-qwen3 (2560d, L17-fredet — bevar model + dimension).
- Ingen WAL på NFS; lockdir lokal disk.
