# Persistence & File Layout

All files live under `storageDir` (plus optional `collection` prefix).

## Files

- `metadata.json`
  - LokiJS database.
  - Stores: `external_id`, `internal_id`, `metadata`, and soft-delete flags.

- `vectors.f32.bin`
  - Raw float32 vectors in a flat binary layout.
  - Vector offset: `internal_id * dim * 4`.

- `dump.bin`
  - WASM ANN snapshot (graph + stored vectors + config header).
  - Loading validates: dim/m/mmax0/ef_construction/max_layers must match.

- `state.json`
  - Small snapshot metadata (`savedAt`, counts, capacity, etc).

- `ann.oplog`
  - Operation log lines:
    - `U <internal_id>` for upsert
    - `D <internal_id>` for delete marker (metadata-only today)
  - Replayed after a successful dump load.

## Save / Load lifecycle

- `open()` initializes and tries `load()`.
- `load()`:
  - tries to load `dump.bin`
  - if dump missing/corrupt and `autoRebuildOnLoad=true`, rebuilds from stored vectors+metadata
  - replays `ann.oplog` to catch up

- `save()`:
  - flush metadata
  - sync vector store
  - writes dump to tmp then atomic rename
  - truncates `ann.oplog` after snapshot

## Rebuild & compaction

- `rebuild({ compact: false })`:
  - reinitializes WASM index
  - scans stored vectors by internal_id and inserts alive items
  - does not rewrite files / does not reorder IDs

- `rebuild({ compact: true })` (default):
  - rewrites `vectors.f32.bin` to contain only alive items
  - rewrites `metadata.json` with new contiguous internal IDs
  - rebuilds WASM index against the compacted store
