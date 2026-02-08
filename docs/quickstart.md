# Quickstart

## Library usage

```ts
import { MiniVectorDB } from "mini-vector-db";

const db = await MiniVectorDB.open({
	storageDir: "./data",
	modelName: "Xenova/all-MiniLM-L6-v2",
	mode: "balanced",
	capacity: 200_000,
	preloadVectors: false,
});

// insert (upsert)
await db.insert({ id: "a", input: "hello", metadata: { tag: "demo" } });

// search
const r1 = await db.search("hello", { topK: 5, score: "similarity" });

// batch search
const r2 = await db.searchMany(["hello", "world"], { topK: 5 });

// soft delete
await db.remove("a");

// rebuild index only (skip deleted)
await db.rebuild({ compact: false, persist: true });

// true compaction rebuild (rewrite files + new contiguous IDs)
await db.rebuild({ compact: true, persist: true });

await db.save();
await db.close();
```

## Inputs

Supported by insert/search:

- `string`
- `Buffer | Uint8Array`
- `number[] | Float32Array`

## Scores

- `l2`: smaller is closer (exact rerank uses float32 L2²)
- `cosine`: derived from normalized vectors
- `similarity`: normalized [0..1]
