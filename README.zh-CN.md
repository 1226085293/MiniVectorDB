# MiniVectorDB —— 本地文件型向量数据库（WASM HNSW + int8 + SIMD + 可选精排）

> 🌍 **语言版本**
>
> - [English](./README.md)
> - **中文**（当前）

MiniVectorDB 是一个面向 **Node.js** 的轻量级、自托管向量数据库：

- 原始向量（**float32，归一化**）落盘存储：`vectors.f32.bin`
- 元数据使用 **LokiJS** 保存：`metadata.json`
- 核心召回为 **WASM（AssemblyScript）实现的 HNSW 类 ANN 索引**，在 WASM 内常驻 **int8 量化向量**
- 可选在候选集上用磁盘 float32 向量做 **精排（exact rerank）**，提升最终质量

适合 “**零基础设施、单机本地、可持久化、可调参**” 的向量检索场景。

---

## 目录

- [特性](#特性)
- [工作原理](#工作原理)
- [安装与构建](#安装与构建)
- [快速开始（库模式）](#快速开始库模式)
- [HTTP API（服务模式）](#http-api服务模式)
- [配置](#配置)
- [持久化与文件结构](#持久化与文件结构)
- [重建与压缩](#重建与压缩)
- [调参指南](#调参指南)
- [文档](#文档)
- [License](#license)

---

## 特性

- **WASM 内 HNSW 类 ANN**：性能稳定、内存可控
- **int8 量化向量常驻 WASM**：相比 JS 侧保留 float32 省内存很多
- **SIMD 加速**：WASM SIMD 路径加速 int8 L2/dot（无 SIMD 自动走标量）
- **可选 float32 精排**：候选集读取 `vectors.f32.bin` 做精确 L2² 排序
- **软删除 + 自动 rebuild**：删除后检索自动过滤；可按阈值触发 rebuild
- **真压缩 compact rebuild**：重写向量/元数据，去掉 deleted 并压缩 internal_id
- **dump + oplog 回放**：`dump.bin` 快速加载；`ann.oplog` 回放追平最近写入
- **离线/预热 embedding 支持**：支持 cacheDir 与 `localFilesOnly`（xenova/transformers）

---

## 工作原理

MiniVectorDB 是典型的“两阶段检索”：

### 1）召回（ANN / WASM）

- 输入（文本/图片/向量）→ embedding 得到 float32
- 对 float32 做 L2 normalize
- 同时生成 int8 量化向量插入 WASM 的 HNSW 图
- 查询在 WASM 内做 HNSW search，得到一批候选 internal_id

### 2）精排（exact / float32）

- 从 `vectors.f32.bin` 读取候选的原始 float32
- 计算精确 L2 distance squared 并排序
- 返回最终 topK

Score 模式：

- `"l2"`：越小越相近（distance²）
- `"cosine"`：越大越相似（由 unit vector 的 L2 推导）
- `"similarity"`：归一化到 [0..1]

---

## 安装与构建

### 运行时要求

- Node.js **18+**（更推荐 Node 20+）
- release.wasm 开启 SIMD 时，需要运行环境支持 WASM SIMD

### 构建

```bash
npm install
npm run build
```

如遇 SIMD 加载问题见：[FAQ](./docs/faq.zh-CN.md)

---

## 快速开始（库模式）

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
		input: "你好世界",
		metadata: { type: "doc" },
	});

	const results = await db.search("你好", { topK: 5, score: "similarity" });
	console.log(results);

	await db.save();
	await db.close();
}

main().catch(console.error);
```

### 支持的输入类型

`insert({ input })` 与 `search(query)` 支持：

- 文本：`string`
- 二进制：`Buffer | Uint8Array`（用于 CLIP 图片/文本或自定义约定）
- 向量：`number[] | Float32Array`

> 注意：同一个库中插入与查询必须使用 **同一个 embedding 模型** 且 **向量维度一致**。

---

## HTTP API（服务模式）

启动：

```bash
npm start
```

支持：

- `POST /insert`
- `POST /search`
- `POST /searchMany`
- `POST /remove`
- `POST /updateMetadata`
- `POST /rebuild`
- `POST /save`
- `GET  /stats`
- `POST /shutdown`（测试/CI 用）

详见：[docs/api.zh-CN.md](./docs/api.zh-CN.md)

---

## 配置

配置优先级：

**显式 opts** > **环境变量** > **默认值/档位 preset**

关键参数：

- `modelName`, `modelArchitecture`（`text` / `clip`）
- `dim`（默认按模型推导：text=384，clip=512）
- `capacity`（internal_id 上限）
- HNSW：`m`, `ef_construction`
- 查询策略：`baseEfSearch`, `rerankMultiplier`, `maxAnnK`, `resultsCap`
- 存储：`storageDir`, `collection`（文件名前缀）, `preloadVectors`
- 重建：`deletedRebuildThreshold`, `autoRebuildOnLoad`

完整列表：[docs/configuration.zh-CN.md](./docs/configuration.zh-CN.md)

---

## 持久化与文件结构

`storageDir` 下：

- `metadata.json`：external_id ↔ internal_id，metadata，deleted 标记
- `vectors.f32.bin`：按 internal*id 连续存 float32（offset = id * dim \_ 4）
- `dump.bin`：WASM ANN dump（图结构 + 向量 + 配置头）
- `state.json`：快照状态信息
- `ann.oplog`：操作日志（upsert/delete）用于加载后回放追平

详见：[docs/persistence.zh-CN.md](./docs/persistence.zh-CN.md)

---

## 重建与压缩

`db.rebuild({ compact })` 或 HTTP `POST /rebuild`

- `compact=false`：只重建 ANN（跳过 deleted），**不重写文件**，internal_id 不变
- `compact=true`（默认）：**真压缩**
  - 重写向量/元数据文件，移除 deleted
  - internal_id 压缩为连续 `0..active-1`
  - 基于新文件重建 ANN

用途：回收磁盘空间、提升连续读写局部性。

更多：[docs/persistence.zh-CN.md#重建与压缩](./docs/persistence.zh-CN.md)

---

## 调参指南

三档 preset：

- `fast`：更低延迟，召回池更小
- `balanced`：默认
- `accurate`：更高召回，更高 CPU/IO

可调项：

- 建库：`m`, `ef_construction`
- 查询：`baseEfSearch`, `rerankMultiplier`, `maxAnnK`, `resultsCap`
- 预加载：`preloadVectors`

详见：[docs/tuning.zh-CN.md](./docs/tuning.zh-CN.md)

---

## 文档

- [快速开始](./docs/quickstart.zh-CN.md)
- [配置](./docs/configuration.zh-CN.md)
- [HTTP API](./docs/api.zh-CN.md)
- [持久化](./docs/persistence.zh-CN.md)
- [Embedding/模型](./docs/embedding.zh-CN.md)
- [示例](./docs/examples.zh-CN.md)
- [FAQ](./docs/faq.zh-CN.md)

---

## License

MIT
