# 示例

## 1）注入自定义 embedder（DI）

```ts
import { MiniVectorDB } from "mini-vector-db";

const myEmbedder = {
	async embed(input: any) {
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

## 2）直接插入向量

```ts
await db.insert({
	id: "v1",
	input: new Float32Array(384),
	metadata: { kind: "vec" },
});
```

> DB 内部会对向量做 normalize（即使你传的是向量）。

## 3）搜索过滤

```ts
const results = await db.search("query", {
	topK: 10,
	filter: { "metadata.tag": "notes" },
});
```

或 predicate：

```ts
const results = await db.search("query", {
	topK: 10,
	filter: (m) => m?.tag === "notes",
});
```

## 4）压缩重建（回收空间）

```ts
await db.rebuild({ compact: true, persist: true });
```
