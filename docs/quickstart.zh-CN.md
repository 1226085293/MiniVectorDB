# 快速开始

## 库模式

```ts
import { MiniVectorDB } from "mini-vector-db";

const db = await MiniVectorDB.open({
	storageDir: "./data",
	modelName: "Xenova/all-MiniLM-L6-v2",
	mode: "balanced",
	capacity: 200_000,
	preloadVectors: false,
});

// 插入（upsert）
await db.insert({ id: "a", input: "你好", metadata: { tag: "demo" } });

// 搜索
const r1 = await db.search("你好", { topK: 5, score: "similarity" });

// 批量搜索
const r2 = await db.searchMany(["你好", "世界"], { topK: 5 });

// 软删除
await db.remove("a");

// 仅重建 ANN（跳过 deleted）
await db.rebuild({ compact: false, persist: true });

// 真压缩：重写文件 + internal_id 连续化
await db.rebuild({ compact: true, persist: true });

await db.save();
await db.close();
```

## 输入类型

insert/search 支持：

- `string`
- `Buffer | Uint8Array`
- `number[] | Float32Array`

## 分数含义

- `l2`：越小越相近（精排使用 float32 L2²）
- `cosine`：由 unit vector 推导
- `similarity`：归一化到 [0..1]
