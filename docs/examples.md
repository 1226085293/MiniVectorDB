# Examples

## 1) Using a custom embedder (DI)

```ts
import { MiniVectorDB } from "mini-vector-db";

const myEmbedder = {
	async embed(input: any) {
		// return Float32Array(dim)
		return new Float32Array(384);
	},
	async init() {},
};

const db = await MiniVectorDB.open({
	storageDir: "./data",
	embedder: myEmbedder,
	dim: 384,
	capacity: 100_000,
});
```

## 2) Insert vectors directly

```ts
await db.insert({
	id: "v1",
	input: new Float32Array(384), // must be normalized or let db normalize (it normalizes)
	metadata: { kind: "vec" },
});
```

## 3) Filter search results

```ts
const results = await db.search("query", {
	topK: 10,
	filter: { "metadata.tag": "notes" }, // or any Loki query object you use
});
```

Or predicate filter:

```ts
const results = await db.search("query", {
	topK: 10,
	filter: (m) => m?.tag === "notes",
});
```

## 4) Compact rebuild (reclaim disk)

```ts
await db.rebuild({ compact: true, persist: true });
```
