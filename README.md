# MiniVectorDB — Local, File-based Vector Database (WASM HNSW + int8 + SIMD + optional rerank)

> 🌍 **Languages**
>
> - **English** (this file)
> - [中文](./README.zh-CN.md)

MiniVectorDB is a lightweight **self-hosted** vector database for **Node.js**:

- Stores **normalized float32 vectors** on disk (`vectors.f32.bin`)
- Keeps metadata in JSON via **LokiJS** (`metadata.json`)
- Uses a **WASM (AssemblyScript) HNSW-like ANN index** for fast recall with **int8 quantized vectors** in memory
- Optionally does **exact rerank** with float32 vectors to improve final quality

It’s designed for “**no infra, single machine, persistent, tunable**” similarity search.

---

## Table of Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Install & build](#install--build)
- [Quickstart (library)](#quickstart-library)
- [HTTP API (server)](#http-api-server)
- [Configuration](#configuration)
- [Persistence & files](#persistence--files)
- [Rebuild & compaction](#rebuild--compaction)
- [Tuning guide](#tuning-guide)
- [Docs](#docs)
- [License](#license)

---

## Features

- **WASM ANN index (HNSW-like)**: predictable memory, fast query
- **int8 quantized vectors in WASM**: much smaller RSS vs keeping float32 in JS memory
- **SIMD acceleration**: WASM SIMD path for int8 L2/dot (fallback to scalar)
- **Optional float32 rerank**: exact L2 on original vectors stored on disk
- **Soft delete + auto rebuild**: deleted items are filtered; rebuild can skip deleted
- **True compaction rebuild**: rewrite vectors/metadata to remove deleted & make internal IDs contiguous
- **Snapshot + oplog replay**: `dump.bin` + metadata + state + `ann.oplog` replay to catch up
- **Offline-friendly embeddings**: cache directory & `localFilesOnly` option for xenova/transformers

---

## How it works

MiniVectorDB is a 2-stage retrieval engine:

1. **Recall (ANN / WASM)**

- Input is embedded to a float32 vector (or provided as vector)
- Vector is L2-normalized
- A quantized **int8** version is inserted into WASM HNSW graph
- Query uses HNSW search in WASM to get candidate `internal_id`s

2. **Rerank (exact / float32)**

- Candidates are read from `vectors.f32.bin`
- Compute exact L2 distance² and sort
- Return final `topK`

Score modes:

- `"l2"`: smaller is closer (distance²)
- `"cosine"`: larger is more similar (derived from L2 for unit vectors)
- `"similarity"`: normalized [0..1] from cosine

---

## Install & build

### Requirements

- Node.js **18+** (Node 20+ recommended)
- A runtime that supports **WASM SIMD** if you use the release build with SIMD enabled

### Build

```bash
npm install
npm run build
```

> If your environment can’t load SIMD WASM, check [FAQ](./docs/faq.md).

---

## Quickstart (library)

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
		input: "hello world",
		metadata: { type: "doc" },
	});

	const results = await db.search("hello", { topK: 5, score: "similarity" });
	console.log(results);

	await db.save();
	await db.close();
}

main().catch(console.error);
```

### Supported input types

`insert({ input })` and `search(query)` accept:

- `string` (text)
- `Buffer | Uint8Array` (binary; used by CLIP image/text or custom embedding)
- `number[] | Float32Array` (vector)

> Important: all vectors in one DB must share the same embedding model and **dimension**.

---

## HTTP API (server)

Start:

```bash
npm start
```

### Endpoints

- `POST /insert`
- `POST /search`
- `POST /searchMany`
- `POST /remove`
- `POST /updateMetadata`
- `POST /rebuild`
- `POST /save`
- `GET  /stats`
- `POST /shutdown` (for tests/CI)

See full API docs: [docs/api.md](./docs/api.md)

---

## Configuration

MiniVectorDB resolves config via:

**explicit options** > **env vars** > **defaults/presets**

Core parameters:

- `modelName`, `modelArchitecture` (`text` or `clip`)
- `dim` (inferred from model/arch by default: text=384, clip=512)
- `capacity` (max internal_id)
- HNSW: `m`, `ef_construction`
- Query strategy: `baseEfSearch`, `rerankMultiplier`, `maxAnnK`, `resultsCap`
- Storage: `storageDir`, `collection` (prefix), `preloadVectors`
- Rebuild: `deletedRebuildThreshold`, `autoRebuildOnLoad`

Full list: [docs/configuration.md](./docs/configuration.md)

---

## Persistence & files

In `storageDir`:

- `metadata.json` (LokiJS): external_id ↔ internal_id, metadata, deleted flags
- `vectors.f32.bin`: raw float32 vectors laid out by internal_id (offset = id _ dim _ 4)
- `dump.bin`: WASM ANN snapshot (graph + vectors + config header)
- `state.json`: small info about snapshot
- `ann.oplog`: operation log (upserts/deletes) replayed after loading dump

Details: [docs/persistence.md](./docs/persistence.md)

---

## Rebuild & compaction

`db.rebuild({ compact })` or HTTP `POST /rebuild`

- `compact=false`: rebuild ANN only (skip deleted), **no file rewrite**, internal IDs unchanged
- `compact=true` (default): **true compaction**
  - rewrite vectors + metadata to remove deleted
  - remap internal IDs to `0..active-1`
  - rebuild ANN from compacted store

This is useful to reclaim disk space and improve locality.

More: [docs/persistence.md#rebuild--compaction](./docs/persistence.md)

---

## Tuning guide

MiniVectorDB ships with 3 presets:

- `fast`: lower latency, smaller candidate pools
- `balanced`: default
- `accurate`: higher recall, more CPU/IO

Plus knobs:

- HNSW build: `m`, `ef_construction`
- Query: `baseEfSearch`, `rerankMultiplier`, `maxAnnK`, `resultsCap`
- Preload vs IO: `preloadVectors`

Guide: [docs/tuning.md](./docs/tuning.md)

---

## Docs

- [Quickstart](./docs/quickstart.md)
- [Configuration](./docs/configuration.md)
- [HTTP API](./docs/api.md)
- [Persistence](./docs/persistence.md)
- [Embedding models](./docs/embedding.md)
- [Examples](./docs/examples.md)
- [FAQ](./docs/faq.md)

---

## License

MIT
