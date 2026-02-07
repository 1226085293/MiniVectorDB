# MiniVectorDB — Local Vector Database (WASM HNSW + SIMD + Optional Re-rank)

> 📖 **Read this document in other languages:**
>
> - **English** (this file)
> - [中文文档](./README.zh-CN.md)

MiniVectorDB is a lightweight, self-hosted vector database for **Node.js**. It uses an **HNSW** (Approximate Nearest Neighbor) index implemented in **WASM (AssemblyScript)**, keeps **int8-quantized vectors** in WASM memory for fast recall, and optionally performs an **exact re-rank** using **float32 original vectors stored on disk** for better accuracy.

It’s a practical choice when you want **zero infrastructure**, **single-machine local persistence**, and a tunable trade-off between **latency / memory / quality**.

---

## Table of Contents

- [Why MiniVectorDB](#why-minivectordb)
- [How It Works](#how-it-works)
- [When to Use (and When Not)](#when-to-use-and-when-not)
- [Install & Build](#install--build)
- [Quick Start (Library Mode)](#quick-start-library-mode)
- [HTTP API (Service Mode)](#http-api-service-mode)
- [Persistence & File Layout](#persistence--file-layout)
- [Environment Variables & Config](#environment-variables--config)
- [Presets & Tuning Tips](#presets--tuning-tips)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)

---

## Why MiniVectorDB

- **WASM HNSW ANN**: core search runs inside WASM with predictable performance characteristics.
- **int8 vectors in memory**: stores quantized vectors in WASM memory to reduce RSS versus float32-in-RAM approaches.
- **Optional float32 re-rank**: recall candidates with HNSW, then compute exact **L2 distance squared** against the original float32 vectors on disk.
- **File-based persistence**: HNSW graph + quantized vectors are stored in `dump.bin`; raw vectors are stored as a contiguous binary file `vectors.f32.bin`.

---

## How It Works

MiniVectorDB is a standard two-stage retrieval pipeline:

### 1) Recall (ANN: WASM + HNSW)

- On insert, MiniVectorDB resolves the input (text/image/vector) into a **float32 vector** and **L2-normalizes** it.
- It also produces an **int8-quantized** copy and stores it inside WASM.
- Queries run HNSW inside WASM and return a set of candidate `internal_id`s (approximate neighbors).

### 2) Re-rank (Exact: float32)

- Original float32 vectors are stored in `vectors.f32.bin`.
- MiniVectorDB loads the float32 vectors for the candidate set, computes **exact L2 distance squared**, re-sorts, and returns topK.

> Key idea:
>
> - Recall speed is dominated by HNSW parameters and candidate pool size
> - Re-rank cost grows with `candidate_count × dim` and IO
> - Larger candidate pools improve recall but increase p95/p99 latency

---

## When to Use (and When Not)

### Use MiniVectorDB when

- You want **local / single-machine** semantic search with **no external vector DB**
- You need better memory efficiency than `number[]`-based in-RAM scans in Node.js
- You prefer an engineering-friendly compromise: **ANN recall + optional exact re-rank**

### Avoid MiniVectorDB when

- You require strict **exact nearest neighbor** for every query
- You need distributed serving, multi-tenant isolation, replication/sharding, etc.
- Your runtime environment cannot reliably support WASM (or SIMD)

---

## Install & Build

```bash
npm install
npm run build
```

### Runtime requirements

- Recommended: Node.js 18+ (Node 20+ is ideal)
- If your Node/WASM runtime does not support WASM SIMD, `release.wasm` may fail to load (see FAQ).

---

## Quick Start (Library Mode)

```ts
import { MiniVectorDB } from "mini-vector-db";

async function main() {
	const db = await MiniVectorDB.open({
		storageDir: "./data",
		modelName: "Xenova/all-MiniLM-L6-v2",
		mode: "balanced",
		capacity: 200_000,
		preloadVectors: false,
	});

	await db.insert({
		id: "doc:1",
		input: "Hello world",
		metadata: { type: "doc" },
	});

	const results = await db.search("Hello", { topK: 5 });
	console.log(results);

	await db.save();
	await db.close();
}

main().catch(console.error);
```

### Supported input types

- `insert({ input })` accepts:
  - text: `string`
  - binary: `Buffer | Uint8Array` (for CLIP or your own convention)
  - vectors: `number[] | Float32Array`

- `search(query)` accepts the same input types.

> Important: the index must be built and queried with the **same embedding model** and the **same vector dimensionality**.

---

## HTTP API (Service Mode)

### Start the server

```bash
npm start
```

### POST `/insert`

```json
{
	"id": "doc:123",
	"input": "any supported input (text/vector/binary)",
	"metadata": { "tag": "notes" }
}
```

### POST `/search`

```json
{
	"query": "any supported input (text/vector/binary)",
	"topK": 10,
	"filter": { "metadata": { "tag": "notes" } }
}
```

### POST `/save`

Persists metadata + dump. (`vectors.f32.bin` is written during inserts; `save` forces sync.)

### GET `/stats`

Returns basic info: mode/model/dim/items/capacity/wasmMaxEf.

### POST `/shutdown`

Graceful shutdown for CI/testing, with a 6-second hard timeout as a fallback.

---

## Persistence & File Layout

Under `storageDir`:

- `metadata.json`
  - LokiJS store for `external_id ↔ internal_id` mapping + metadata

- `vectors.f32.bin`
  - contiguous float32 vector store
  - offset = `internal_id * dim * 4`

- `dump.bin`
  - WASM HNSW dump (graph structure + int8 vectors)

Load behavior:

- `MiniVectorDB.open()` runs `init()` and tries `load()` (missing dump is ignored)
- `load()` optionally preloads `vectors.f32.bin` if `preloadVectors` is enabled

---

## Environment Variables & Config

Your code uses two groups of environment variables:

1. **Server settings** (Fastify startup)
2. **DB settings** (open/init/search strategy)

> Precedence (DB settings):
> explicit `opts` > environment variables > defaults / preset inference

### 1) Server environment variables (`src/api/server.ts`)

| Env Var                  | Purpose              |                   Default | When it applies | Notes                                             |          |           |
| ------------------------ | -------------------- | ------------------------: | --------------- | ------------------------------------------------- | -------- | --------- |
| `API_HOST`               | bind address         |               `127.0.0.1` | startup         | use `0.0.0.0` for containers/LAN                  |          |           |
| `API_PORT`               | server port          |                    `3000` | startup         |                                                   |          |           |
| `MINIVECTOR_STORAGE_DIR` | DB data directory    |      `process.cwd()/data` | startup         | passed to `MiniVectorDB.open({ storageDir })`     |          |           |
| `MODEL_NAME`             | embedding model      | `Xenova/all-MiniLM-L6-v2` | startup         | affects arch/dim inference                        |          |           |
| `MINIVECTOR_MODE`        | preset mode          |                `balanced` | startup         | `fast                                             | balanced | accurate` |
| `HNSW_CAPACITY`          | capacity limit       |               `1_200_000` | startup         | must be big enough for your dataset               |          |           |
| `PRELOAD_VECTORS`        | preload vectors file |                       `0` | startup         | `1` to enable (higher RSS, lower IO tail latency) |          |           |

### 2) Library environment variables (`src/index.ts` → `resolveOpenConfig`)

| Env Var                  | Config field       | Purpose                   |                        Default | When it applies               | Requires rebuild                  |
| ------------------------ | ------------------ | ------------------------- | -----------------------------: | ----------------------------- | --------------------------------- |
| `MINIVECTOR_STORAGE_DIR` | `storageDir`       | data directory            |                       `./data` | on open                       | no                                |
| `MODEL_NAME`             | `modelName`        | embedding model           |      `Xenova/all-MiniLM-L6-v2` | on open                       | **usually yes**                   |
| `MINIVECTOR_MODE`        | `mode`             | preset mode               |                     `balanced` | on open                       | depends (see below)               |
| `VECTOR_DIM`             | `dim`              | vector dimension          | inferred (text=384 / clip=512) | open/init                     | **yes**                           |
| `HNSW_CAPACITY`          | `capacity`         | internal_id upper bound   |                    `1_200_000` | init/insert                   | not necessarily (see note)        |
| `PRELOAD_VECTORS`        | `preloadVectors`   | preload `vectors.f32.bin` |                            `0` | load                          | no                                |
| `HNSW_M`                 | `m`                | HNSW M                    |                         preset | init                          | **yes** (`dump.bin` not reusable) |
| `HNSW_EF`                | `ef_construction`  | build efConstruction      |                         preset | init                          | **yes** (`dump.bin` not reusable) |
| `BASE_EF_SEARCH`         | `baseEfSearch`     | efSearch baseline         |                         preset | query-time (dynamic)          | no                                |
| `RERANK_MULTIPLIER`      | `rerankMultiplier` | annK = topK × multiplier  |                         preset | query-time                    | no                                |
| `MAX_ANN_K`              | `maxAnnK`          | annK upper bound          |                         preset | query-time                    | no                                |
| `HNSW_RESULTS_CAP`       | `resultsCap`       | WASM result buffer cap    |                         preset | init / may grow at query-time | no (but limited by WASM MAX_EF)   |

#### Practical “rebuild required?” rules

- **Rebuild required / old `dump.bin` cannot be reused**: `dim`, `m`, `ef_construction`, embedding model changes
- **No rebuild (runtime-tunable)**: `baseEfSearch`, `rerankMultiplier`, `maxAnnK`, `resultsCap`, `preloadVectors`
- `capacity`: increasing it doesn’t inherently require rebuilding, but you should keep a consistent storage layout and plan capacity up front.

---

## Presets & Tuning Tips

### Preset overview

- `fast`
  - faster build, smaller candidate pool, low latency, lower recall

- `balanced`
  - recommended default: better trade-off

- `accurate`
  - higher recall target; avoid overly large candidate pools that inflate p95/p99

### Your query-time auto strategy

- `efSearch = max(baseEfSearch, topK * 2)`
- `annK = min(topK * rerankMultiplier, maxAnnK, wasmMaxEf)`
- `resultsCap` grows automatically until it’s ≥ `annK`

#### What to tune first

- Want more accuracy: increase `RERANK_MULTIPLIER` or `BASE_EF_SEARCH`
- Want lower latency: decrease `RERANK_MULTIPLIER` first, then `BASE_EF_SEARCH`
- Want better index quality: increase `m` / `ef_construction` (requires rebuild)

---

## FAQ / Troubleshooting

### 1) Will rebuilding the index “damage” existing data?

Not magically, but your storage consists of three parts:

- `metadata.json`: id mapping + metadata
- `vectors.f32.bin`: float32 vectors
- `dump.bin`: HNSW graph + quantized vectors (strongly coupled to build parameters)

If you change `dim / m / ef_construction` or switch embedding models:

- **old `dump.bin` is not reusable** (remove/backup and rebuild)
- `vectors.f32.bin` may still exist, but if dim/model changed, those vectors are no longer semantically compatible—re-embed/rebuild is recommended.

Safe workflow:

- backup `storageDir`
- remove `dump.bin`
- rebuild from source data (or reinsert)

---

### 2) “It loads but returns nothing” / results look wrong

Common causes:

- index was built with a different embedding model than you’re querying with
- dimension mismatch between configured `dim` and actual vectors
- mixing “raw vectors” inserts with “text embedding” queries from a different model

Check:

- `/stats` (model/dim/mode)
- ensure both insert and search use the same embedding source

---

### 3) Error: `Vector dimension mismatch`

Fix:

- for vector inputs: ensure length equals `dim`
- for text/binary inputs: ensure the embedder output dimension matches `dim`
  - you can avoid manual `VECTOR_DIM` and rely on inference from `modelName`

---

### 4) Error: `Database capacity exceeded / overflow`

This means your `internal_id` reached the hard limit `capacity`.
Fix:

- reopen with a larger `capacity` (plan 2×–4× headroom)
- rebuild or migrate storage as needed

---

### 5) Why don’t values above 4096 for `maxAnnK/resultsCap` work?

Your WASM layer has a hard limit (MAX_EF=4096). Therefore:

- `annK` is clamped by `wasmMaxEf`
- `resultsCap` above that is pointless

Recommendation:

- document the 4096 cap
- keep presets within 4096

---

### 6) High p95/p99 latency — how do I reduce it?

Usually the candidate pool is too large.
Steps:

1. lower `RERANK_MULTIPLIER`
2. lower `BASE_EF_SEARCH`
3. enable `PRELOAD_VECTORS=1` to reduce IO tail latency (higher RSS)

---

### 7) Why is `score` smaller = more similar?

MiniVectorDB returns **L2 distance squared** after exact re-rank:

- **smaller is better**

This differs from “cosine similarity larger is better” APIs. Since vectors are normalized, L2 and cosine are monotonic-related, but MiniVectorDB keeps L2^2 for simplicity and speed.

---

### 8) What does `PRELOAD_VECTORS=1` do?

- loads `vectors.f32.bin` into RAM at `load()` time
- re-rank reads from memory, reducing random disk IO
- increases RSS proportional to vector file size

---

### 9) WASM SIMD load failure

Likely:

- Node too old
- runtime lacks WASM SIMD support

Fix:

- upgrade Node (18+/20+)
- provide a non-SIMD wasm build or fallback strategy and document it

---

### 10) Changing `MINIVECTOR_MODE` doesn’t seem to change anything

Possible reasons:

- env vars override the preset values (e.g., `HNSW_M`, `HNSW_EF`, `RERANK_MULTIPLIER`)
- you changed build params but didn’t rebuild (still using old `dump.bin`)

Debug:

- log `db.cfg`
- verify the final resolved config source (opts/env/preset)
- rebuild after changing build params

---

## Project Structure

- `assembly/`: AssemblyScript HNSW + distance kernels + allocator
- `src/core/wasm-bridge.ts`: WASM bridge, efSearch/resultsCap control, dump IO
- `src/storage/meta-db.ts`: LokiJS metadata store (external_id ↔ internal_id + metadata)
- `src/embedder.ts`: local embedding (`@xenova/transformers`)
- `src/index.ts`: core MiniVectorDB logic (open/insert/search/save/load)
- `src/api/server.ts`: Fastify HTTP API

---

## License

MIT
