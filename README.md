# MiniVectorDB (WASM + HNSW, SIMD-optimized)

A lightweight, self-hosted vector database built with **Node.js + AssemblyScript (WASM SIMD)**.  
It uses an **HNSW** approximate nearest neighbor index for fast retrieval, plus an optional **exact re-rank** stage using stored float vectors.

- Default language: **English**
- 中文版：see **[README.zh-CN.md](./README.zh-CN.md)**

## What is this?

If you’ve never used a “vector database” before, think of it as:

> Turn data (text / image / audio / etc.) into **vectors** (arrays of numbers).  
> Similar items have vectors that are “close” to each other.  
> A vector DB lets you quickly find the closest items to a query vector.

This project provides:

- A **library** (`MiniVectorDB`) you can use in Node.js
- A simple **HTTP API server** (Fastify) with `/insert` and `/search`

---

## Features

- **HNSW index in WASM (AssemblyScript)** for fast approximate search
- **SIMD-accelerated** int8 L2 distance inside WASM (when supported)
- **Int8 quantized vectors** in WASM for speed/memory
- **Float32 vector store on disk** for exact L2 re-ranking (better accuracy)
- **Metadata store** via LokiJS (JSON on disk)
- **Persistence**
  - `dump.bin` stores the HNSW graph + quantized vectors
  - `vectors.f32.bin` stores float vectors
  - `metadata.json` stores external_id ↔ internal_id + metadata

---

## Suitable environments

### Runtime

- **Node.js 18+ recommended** (WASM SIMD support is expected)
- macOS / Linux / Windows

### Build tools

- `npm` (or pnpm/yarn if you adapt scripts)

### Hardware

- Any CPU works, but SIMD-capable CPUs improve performance.

> ⚠️ If your Node/WASM runtime does not support WASM SIMD, `build/release.wasm` may fail to instantiate.

---

## Install & Build

```bash
npm install
npm run build
```

### Run the API server

```bash
npm start
```

Server entry: `src/api/server.ts`

---

## Configuration (.env)

The server loads `.env` from the repo root.

Common settings:

| Env var            | Meaning                                                                     | Default                   |
| ------------------ | --------------------------------------------------------------------------- | ------------------------- |
| `API_HOST`         | bind host                                                                   | `127.0.0.1`               |
| `API_PORT`         | server port                                                                 | `3000`                    |
| `MODEL_NAME`       | embedding model name used by library (not required for API numeric vectors) | `Xenova/all-MiniLM-L6-v2` |
| `VECTOR_DIM`       | vector dimension                                                            | `384`                     |
| `HNSW_M`           | HNSW M (neighbors per layer)                                                | `16`                      |
| `HNSW_EF`          | `efConstruction`                                                            | `100`                     |
| `HNSW_EF_SEARCH`   | `efSearch`                                                                  | `50`                      |
| `HNSW_CAPACITY`    | max elements reserved in index                                              | `1200000`                 |
| `HNSW_RESULTS_CAP` | how many raw ANN candidates WASM returns at most                            | `1000`                    |
| `MAX_ANN_K`        | cap for ANN candidates in library stage                                     | `10000`                   |
| `HNSW_SEED`        | RNG seed for level assignment                                               | auto                      |

### Dimension tips

- If you use `Xenova/all-MiniLM-L6-v2` → DIM is usually **384**
- If you use CLIP `Xenova/clip-vit-base-patch32` → DIM is usually **512**

> The WASM config expects `DIM` to be a multiple of 4.

---

## Data files

By default the API uses:

- `data/dump.bin` (HNSW dump)
- `data/vectors.f32.bin` (Float32 vector store)
- `data/metadata.json` (LokiJS metadata)

On startup, server tries to load `dump.bin`. If load fails, it starts fresh.

---

## HTTP API Guide (for people new to vector databases)

### 1) You need vectors (embeddings) first

This server’s `/insert` and `/search` expect **numeric vectors** (`number[]`) with length `VECTOR_DIM`.

How do you get vectors?

- Use an embedding model (OpenAI / local models / Xenova transformers, etc.)
- The important rule: **insert vectors and query vectors must come from the same model + same dimension**

If you don’t have embeddings yet, you can:

- Generate them in your app and call this API, or
- Use this project as a library (it can embed text/images locally via `@xenova/transformers`)

---

### 2) Insert

**POST** `/insert`

Body:

```json
{
  "id": "doc:123",
  "vector": [0.01, -0.02, ...],
  "metadata": { "title": "Hello", "tag": "notes" }
}
```

Example (curl):

```bash
curl -X POST http://127.0.0.1:3000/insert \
  -H "content-type: application/json" \
  -d '{
    "id": "doc:123",
    "vector": [0,0,0,0 /* ... must be VECTOR_DIM numbers ... */],
    "metadata": { "type": "doc", "lang": "en" }
  }'
```

Notes:

- `id` is your external identifier (string)
- If you insert the same `id` again, it updates the stored vector and **reconnects** neighbors in HNSW.

---

### 3) Search

**POST** `/search`

Body:

```json
{
  "vector": [0.01, -0.02, ...],
  "k": 10,
  "filter": { /* optional */ }
}
```

Example:

```bash
curl -X POST http://127.0.0.1:3000/search \
  -H "content-type: application/json" \
  -d '{
    "vector": [0,0,0,0 /* ... VECTOR_DIM ... */],
    "k": 5
  }'
```

Response:

```json
{
	"results": [{ "id": "doc:123", "score": 0.42, "metadata": { "type": "doc" } }]
}
```

How to interpret `score`:

- It is **L2 distance squared** (after exact re-rank using Float32 vectors)
- **Smaller = more similar**

#### Optional filter

`filter` is passed to LokiJS `find()` internally. Practically, you’ll filter on stored `metadata`.

Example idea (your exact query shape may vary depending on how you store metadata):

```json
{
	"filter": { "metadata": { "type": "doc" } }
}
```

---

### 4) Save index to disk

**POST** `/save`

```bash
curl -X POST http://127.0.0.1:3000/save
```

This writes:

- `data/dump.bin` (HNSW graph dump)
- metadata is already autosaved; vectors are flushed to disk

---

### 5) Stats

**GET** `/stats`

```bash
curl http://127.0.0.1:3000/stats
```

Returns number of items, paths, capacity, etc.

---

### 6) Shutdown (testing/CI)

**POST** `/shutdown`

```bash
curl -X POST http://127.0.0.1:3000/shutdown
```

The server replies first, then closes DB and exits (with a hard timeout fallback).

---

## Library usage (Node.js)

If you don’t want to run an HTTP service, you can use the library directly.
The library can accept:

- vectors (`number[]` / `Float32Array`)
- or raw inputs (`string` / `Buffer` / `Uint8Array`) and embed locally via `@xenova/transformers`

Example:

```ts
import { MiniVectorDB } from "mini-vector-db";

const db = new MiniVectorDB({
	dim: 384,
	m: 16,
	ef_construction: 100,
	ef_search: 50,
	capacity: 200_000,
	resultsCap: 1000,
	modelName: "Xenova/all-MiniLM-L6-v2",
	modelArchitecture: "text",
});

await db.init();

await db.insert({
	id: "doc:1",
	vector: "Hello world", // can be text (will embed)
	metadata: { type: "doc" },
});

const results = await db.search("hello", 5);
console.log(results);

await db.save("./data/dump.bin");
await db.close();
```

---

## Performance & capacity notes

- `HNSW_CAPACITY` is a **hard limit**. If you exceed it, inserts will throw.
- Disk usage for float vectors:
  - `capacity * dim * 4 bytes`
  - Example: `1,200,000 * 384 * 4 ≈ 1.84 GB`

- RAM usage depends on:
  - quantized vectors (`dim` bytes each)
  - HNSW node graph (neighbors per node, level distribution)
  - WASM memory growth

For large capacities, plan for **multiple GB** disk and **significant RAM**.

---

## Project structure

- `assembly/` — AssemblyScript HNSW + SIMD distance + custom allocator
- `src/core/wasm-bridge.ts` — calls into WASM, manages scratch buffers, dump load/save
- `src/storage/meta-db.ts` — LokiJS metadata mapping
- `src/embedder.ts` — local embedding via `@xenova/transformers`
- `src/api/server.ts` — Fastify HTTP API
