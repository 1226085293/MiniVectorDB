# Configuration

Config is resolved as:

**explicit open() options** > **env vars** > **defaults/presets**

## Open options (DBOpenOptions)

Common fields:

- `storageDir?: string`  
  Directory for all DB files.

- `collection?: string`  
  Optional prefix: files become `${collection}.metadata.json`, etc.

- `modelName?: string`  
  Default: `Xenova/all-MiniLM-L6-v2`

- `modelArchitecture?: "text" | "clip"`  
  Auto-inferred from `modelName` if not provided.

- `dim?: number`  
  Default: inferred by architecture (text=384, clip=512).  
  Must match actual embedder output.

- `mode?: "fast" | "balanced" | "accurate"`  
  Preset for ANN/search strategy.

- `capacity?: number`  
  Maximum internal_id. Default: `1_200_000`.

- `preloadVectors?: boolean`  
  If true, load `vectors.f32.bin` into memory for faster rerank (uses more RAM).

- `m?: number` / `ef_construction?: number`  
  HNSW build parameters. Changing them typically requires rebuilding (dump not reusable).

- `embeddingCacheSize?: number`  
  Text embedding cache size in memory (LRU-like behavior).

- `deletedRebuildThreshold?: number`  
  Auto rebuild threshold: when deletedSinceRebuild / total >= threshold (default 0.2).

- `autoRebuildOnLoad?: boolean`  
  If dump missing/corrupt, rebuild from stored vectors+metadata (default true).

- `modelCacheDir?: string` / `localFilesOnly?: boolean`  
  For xenova/transformers offline behavior.

## Environment variables

These are supported in `src/config.ts` and `src/api/server.ts`:

### Server

- `API_HOST` (default `127.0.0.1`)
- `API_PORT` (default `3000`)
- `MINIVECTOR_STORAGE_DIR`
- `MINIVECTOR_COLLECTION`
- `MODEL_NAME`
- `MINIVECTOR_MODE`
- `HNSW_CAPACITY`
- `PRELOAD_VECTORS`

### DB

- `MINIVECTOR_STORAGE_DIR`
- `MINIVECTOR_COLLECTION`
- `MODEL_NAME`
- `MINIVECTOR_MODE`
- `VECTOR_DIM`
- `HNSW_CAPACITY`
- `PRELOAD_VECTORS`
- `HNSW_M`
- `HNSW_EF`
- `BASE_EF_SEARCH`
- `RERANK_MULTIPLIER`
- `MAX_ANN_K`
- `HNSW_RESULTS_CAP`
- `MINIVECTOR_DELETED_REBUILD_THRESHOLD`
- `MINIVECTOR_AUTO_REBUILD_ON_LOAD`
- `MINIVECTOR_MODEL_CACHE_DIR`
- `MINIVECTOR_LOCAL_FILES_ONLY`
- `EMBEDDING_CACHE_SIZE`

## Presets

- `fast`: lower latency, smaller pools
- `balanced`: default
- `accurate`: higher recall, more CPU/IO
