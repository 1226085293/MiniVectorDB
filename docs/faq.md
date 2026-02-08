# FAQ

## 1) WASM SIMD load fails

If your Node/WASM runtime doesn’t support SIMD, a SIMD-enabled `release.wasm` may fail to instantiate.

Options:

- use a runtime that supports WASM SIMD (Node 18/20 typically do)
- build a non-SIMD WASM target (if you maintain multiple targets)
- run in an environment where SIMD is enabled

## 2) “Vector dimension mismatch”

All vectors must have the same `dim` as the configured DB.
If you change model or architecture, you must rebuild or use a new storageDir.

## 3) Disk grows after deletes

Deletes are soft. To reclaim disk:

- `db.rebuild({ compact: true })`

## 4) How to increase capacity

Capacity is the maximum internal_id. If exceeded:

- rebuild with a larger capacity
  - `db.rebuild({ capacity: old*2, compact: false })`
  - or compaction rebuild `compact: true` (also rewrites files)
