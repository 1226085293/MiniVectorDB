# Tuning guide

MiniVectorDB uses:

- WASM HNSW-like ANN for recall
- Optional float32 exact rerank for final ranking

## Presets

- `fast`
  - lower `m`, moderate `ef_construction`
  - smaller `baseEfSearch` / `rerankMultiplier`
  - lower latency, lower recall

- `balanced` (default)
  - good overall tradeoff

- `accurate`
  - larger `m`, much larger `ef_construction`
  - bigger candidate pools
  - higher recall, more CPU/IO

## Key knobs

### Build-time (changes usually invalidate dumps)

- `dim`
- `m`
- `ef_construction`
- model choice / architecture

### Query-time (safe to adjust without changing on-disk layout)

- `baseEfSearch`: baseline efSearch (code clamps efSearch >= topK\*2)
- `rerankMultiplier`: candidates = topK \* multiplier (then capped by maxAnnK)
- `maxAnnK`: candidate pool ceiling
- `resultsCap`: WASM result buffer cap (clamped to wasm MAX_EF=4096)
- `preloadVectors`: trade RAM for less IO during rerank

## Practical tips

- Start with `balanced`.
- If recall is unstable: raise `baseEfSearch` and `rerankMultiplier`.
- If p95 IO spikes: reduce `rerankMultiplier` or enable `preloadVectors` (if memory allows).
- For large datasets: plan capacity ahead; compaction rebuild can reclaim disk and improve locality.
