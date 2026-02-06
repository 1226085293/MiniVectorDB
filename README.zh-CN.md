# MiniVectorDB（WASM + HNSW，SIMD 优化）

一个轻量级、自托管的向量数据库，基于 **Node.js + AssemblyScript（WASM SIMD）**。  
内部使用 **HNSW**（近似最近邻）进行快速召回，并可用磁盘中的 Float32 原始向量进行 **精排（re-rank）** 提升准确率。

- 默认展示：**英文 README（README.md）**
- 中文版：本文件 **README.zh-CN.md**

---

## 这是什么？（给没用过向量数据库的程序员）

你可以把“向量数据库”理解为：

> 把数据（文本/图片/音频等）转换成 **向量**（一串数字）。  
> 相似的数据在向量空间里“距离更近”。  
> 向量 DB 就是让你能快速找出“最相似的那些条目”。

本项目提供：

- **Node.js 库**：`MiniVectorDB`
- **HTTP API 服务**（Fastify）：`/insert`、`/search` 等

---

## 特性

- **WASM（AssemblyScript）实现 HNSW**，近似检索速度快
- **SIMD 加速**的 int8 L2 距离计算（取决于运行时是否支持 WASM SIMD）
- **WASM 内部存 int8 量化向量**：更省内存、更快
- **磁盘存 Float32 原始向量**：用于精排，结果更准
- **LokiJS 存元数据**：JSON 文件落盘
- **持久化**
  - `dump.bin`：保存 HNSW 图结构 + 量化向量
  - `vectors.f32.bin`：保存 Float32 向量
  - `metadata.json`：保存 external_id ↔ internal_id + metadata

---

## 适用环境

### 运行时

- **建议 Node.js 18+**（预期支持 WASM SIMD）
- macOS / Linux / Windows

### 构建工具

- `npm`（或自行适配 pnpm/yarn）

> ⚠️ 如果你的 Node/WASM 运行时不支持 WASM SIMD，`build/release.wasm` 可能无法加载。

---

## 安装与构建

```bash
npm install
npm run build
```

### 启动 API 服务

```bash
npm start
```

服务入口：`src/api/server.ts`

---

## 配置（.env）

服务会从项目根目录读取 `.env`。

常用变量：

| 环境变量           | 含义                                             | 默认值                    |
| ------------------ | ------------------------------------------------ | ------------------------- |
| `API_HOST`         | 绑定地址                                         | `127.0.0.1`               |
| `API_PORT`         | 端口                                             | `3000`                    |
| `MODEL_NAME`       | 库模式用的嵌入模型（API 直接传数值向量时不必用） | `Xenova/all-MiniLM-L6-v2` |
| `VECTOR_DIM`       | 向量维度                                         | `384`                     |
| `HNSW_M`           | HNSW 的 M（每层邻居数）                          | `16`                      |
| `HNSW_EF`          | 构建阶段 `efConstruction`                        | `100`                     |
| `HNSW_EF_SEARCH`   | 查询阶段 `efSearch`                              | `50`                      |
| `HNSW_CAPACITY`    | 索引容量上限                                     | `1200000`                 |
| `HNSW_RESULTS_CAP` | WASM 最多返回多少 ANN 候选                       | `1000`                    |
| `MAX_ANN_K`        | 库层 ANN 候选上限                                | `10000`                   |
| `HNSW_SEED`        | 随机种子（用于节点层级）                         | 自动                      |

### 维度提示

- `Xenova/all-MiniLM-L6-v2` 常见维度：**384**
- CLIP `Xenova/clip-vit-base-patch32` 常见维度：**512**

> WASM 侧要求 `DIM` 是 4 的倍数（便于 SIMD 优化）。

---

## 数据文件说明

API 默认使用：

- `data/dump.bin`（HNSW dump）
- `data/vectors.f32.bin`（Float32 向量文件）
- `data/metadata.json`（LokiJS 元数据）

服务启动时会尝试加载 `dump.bin`；加载失败则从空库开始。

---

## HTTP API 使用指南（面向不了解向量数据库的人）

### 1）你需要先有“向量”（embedding）

本服务的 `/insert`、`/search` 都要求传入 **数值向量**（`number[]`），且长度必须等于 `VECTOR_DIM`。

向量怎么来？

- 用 embedding 模型生成（OpenAI / 本地模型 / Xenova transformers 等）
- 关键规则：**插入用的向量和查询用的向量必须来自同一个模型，且维度一致**

如果你还没有 embedding：

- 可以在你的业务服务里先生成向量，再调用本 API
- 或直接使用本项目的 **库模式**（可在本地 embed 文本/图片）

---

### 2）写入（Insert）

**POST** `/insert`

请求体：

```json
{
  "id": "doc:123",
  "vector": [0.01, -0.02, ...],
  "metadata": { "title": "你好", "tag": "notes" }
}
```

curl 示例：

```bash
curl -X POST http://127.0.0.1:3000/insert \
  -H "content-type: application/json" \
  -d '{
    "id": "doc:123",
    "vector": [0,0,0,0 /* ... 必须是 VECTOR_DIM 个数字 ... */],
    "metadata": { "type": "doc", "lang": "zh" }
  }'
```

说明：

- `id` 是你的外部 ID（字符串）
- 如果重复插入同一个 `id`，会更新向量并进行图的邻居重连（reconnect）

---

### 3）查询（Search）

**POST** `/search`

请求体：

```json
{
  "vector": [0.01, -0.02, ...],
  "k": 10,
  "filter": { /* 可选 */ }
}
```

curl 示例：

```bash
curl -X POST http://127.0.0.1:3000/search \
  -H "content-type: application/json" \
  -d '{
    "vector": [0,0,0,0 /* ... VECTOR_DIM ... */],
    "k": 5
  }'
```

返回：

```json
{
	"results": [{ "id": "doc:123", "score": 0.42, "metadata": { "type": "doc" } }]
}
```

如何理解 `score`：

- 这是 **L2 距离的平方**（使用磁盘 Float32 向量精排后的结果）
- **越小越相似**

#### 可选 filter

`filter` 会传给 LokiJS 的 `find()`，通常用于按 `metadata` 过滤。

示意（具体查询结构取决于你 metadata 的存法）：

```json
{
	"filter": { "metadata": { "type": "doc" } }
}
```

---

### 4）保存索引

**POST** `/save`

```bash
curl -X POST http://127.0.0.1:3000/save
```

会写入：

- `data/dump.bin`（HNSW dump）
- 向量文件已落盘并 sync；元数据 LokiJS 默认会自动保存

---

### 5）查看统计信息

**GET** `/stats`

```bash
curl http://127.0.0.1:3000/stats
```

返回条目数、路径、容量等信息。

---

### 6）关闭服务（测试/CI）

**POST** `/shutdown`

```bash
curl -X POST http://127.0.0.1:3000/shutdown
```

会先返回响应，再异步关闭数据库并退出（有硬超时兜底，避免 CI 卡住）。

---

## 作为库使用（Node.js）

如果你不想起 HTTP 服务，可以直接使用库。
库支持传入：

- 数值向量（`number[]` / `Float32Array`）
- 或原始输入（`string` / `Buffer` / `Uint8Array`），在本地用 `@xenova/transformers` 生成 embedding

示例：

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
	vector: "你好世界", // 直接传文本（会 embed）
	metadata: { type: "doc" },
});

const results = await db.search("你好", 5);
console.log(results);

await db.save("./data/dump.bin");
await db.close();
```

---

## 性能与容量提示

- `HNSW_CAPACITY` 是 **硬上限**，超过会报错（需要调大容量）
- Float32 向量磁盘占用：
  - `capacity * dim * 4 bytes`
  - 例如：`1,200,000 * 384 * 4 ≈ 1.84 GB`

- 内存占用与以下有关：
  - WASM 内的量化向量（每条 `dim` 字节）
  - HNSW 图结构（邻居数量、层级分布）
  - WASM 内存增长策略

大容量下请预留 **多 GB 磁盘** 和 **较高 RAM**。

---

## 项目结构

- `assembly/`：AssemblyScript HNSW + SIMD 距离 + 自定义 bump allocator
- `src/core/wasm-bridge.ts`：WASM 调用桥接、scratch buffer、dump 读写
- `src/storage/meta-db.ts`：LokiJS 元数据映射
- `src/embedder.ts`：本地嵌入（`@xenova/transformers`）
- `src/api/server.ts`：Fastify HTTP API
