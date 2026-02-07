# MiniVectorDB —— 本地文件型向量数据库（WASM HNSW + SIMD + 可选精排）

> 📖 **其他语言版本：**
>
> - [English](./README.md)
> - **中文**（当前）

MiniVectorDB 是一个面向 **Node.js** 的轻量级、自托管向量数据库：核心 ANN 索引使用 **WASM（AssemblyScript）实现的 HNSW**，并在 WASM 内部用 **int8 量化向量**进行快速召回；随后可基于磁盘上的 **float32 原始向量**进行 **精排（re-rank）**，以获得更高的准确率。

它适合希望 “**零基础设施、单机本地、可持久化、可调参**” 的向量检索场景。

---

## 目录

- [为什么是 MiniVectorDB](#为什么是-minivectordb)
- [工作原理](#工作原理)
- [适用场景与不适用场景](#适用场景与不适用场景)
- [安装与构建](#安装与构建)
- [快速开始（库模式）](#快速开始库模式)
- [HTTP API（服务模式）](#http-api服务模式)
- [持久化与文件结构](#持久化与文件结构)
- [环境变量与配置项](#环境变量与配置项)
- [档位与调参建议](#档位与调参建议)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [项目结构](#项目结构)
- [License](#license)

---

## 为什么是 MiniVectorDB

- **WASM 内 HNSW ANN**：核心检索在 WASM 中完成，性能与内存更可控。
- **int8 量化向量常驻内存**：相比 float32 全量常驻内存，常见情况下能显著降低 RSS。
- **可选 float32 精排**：ANN 先召回候选，再用磁盘原始向量精确 L2 重新排序，提升 recall/ndcg。
- **文件型持久化**：索引结构以二进制 `dump.bin` 存储；原始向量以连续二进制 `vectors.f32.bin` 存储，适合大规模数据落盘。

---

## 工作原理

MiniVectorDB 是一个典型的“两阶段检索”：

### 1）召回阶段（ANN：WASM + HNSW）

- 插入时将输入（文本/图片/向量）解析成 **float32 向量**并归一化（L2 normalize）
- 同时生成 **int8 量化向量**写入 WASM 内存
- WASM 内使用 HNSW 搜索，得到一批候选 internal_id（近似最近邻）

### 2）精排阶段（rerank：float32）

- 原始 float32 向量存储在 `vectors.f32.bin`
- 将 ANN 的候选集合对应向量读出，计算 **精确 L2 distance squared** 并重新排序
- 返回最终 topK（score 越小越相似）

> 设计要点：
>
> - 召回速度由 HNSW 与候选池大小决定
> - 精排成本主要是候选池大小 × dim 的 float32 计算与 IO
> - 候选池越大，召回越稳，但 p95/p99 更容易上升

---

## 适用场景与不适用场景

### 适用 MiniVectorDB

- 本地应用、桌面端、边缘端、单机服务：希望 **不依赖外部向量库/服务**
- 数据规模从几万到几十万甚至更多，希望避免 JS `number[]` 全量常驻内存导致的 RSS 膨胀
- 希望在 “速度 / 内存 / 准确率” 之间做工程折中（ANN + rerank）

### 不适用 MiniVectorDB

- 你需要严格 exact NN（每次查询都必须 100% 等同于暴力）
- 你需要分布式、网络服务、多租户隔离、权限、复制、横向扩展等数据库级能力
- 运行环境不能稳定支持 WASM（或 SIMD）

---

## 安装与构建

```bash
npm install
npm run build
```

### 运行时要求

- 推荐 Node.js 18+（Node 20+ 更理想）
- 如果你的 Node/WASM 运行时不支持 SIMD，`release.wasm` 可能加载失败（见 FAQ）

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

	const results = await db.search("你好", { topK: 5 });
	console.log(results);

	await db.save();
	await db.close();
}

main().catch(console.error);
```

### 输入类型

- `insert({ input })` 支持：
  - 文本：`string`
  - 二进制：`Buffer | Uint8Array`（用于 clip 或你自己的约定）
  - 向量：`number[] | Float32Array`

- `search(query)` 支持同样类型

> 注意：同一索引中插入与查询必须使用 **同一个 embedding 模型** 且 **维度一致**。

---

## HTTP API（服务模式）

### 启动

```bash
npm start
```

### POST `/insert`

```json
{
	"id": "doc:123",
	"input": "任意支持的输入（文本/向量/二进制）",
	"metadata": { "tag": "notes" }
}
```

### POST `/search`

```json
{
	"query": "任意支持的输入（文本/向量/二进制）",
	"topK": 10,
	"filter": { "metadata": { "tag": "notes" } }
}
```

### POST `/save`

保存 metadata + dump + 向量文件（向量文件写入在 insert 时完成，save 会强制 sync）。

### GET `/stats`

返回基础统计信息（mode/model/dim/items/capacity/wasmMaxEf）。

### POST `/shutdown`

用于 CI/测试的优雅退出（带 6 秒硬超时兜底）。

---

## 持久化与文件结构

在 `storageDir` 目录下：

- `metadata.json`
  - LokiJS 存储 external_id ↔ internal_id 以及 metadata

- `vectors.f32.bin`
  - 连续的 float32 向量文件（按 internal*id 定位：offset = id * dim \_ 4）

- `dump.bin`
  - WASM HNSW 的二进制 dump（图结构 + int8 量化向量）

启动加载逻辑：

- `MiniVectorDB.open()` 会 `init()` 并尝试 `load()`（dump 不存在则忽略）
- `load()` 成功后可按需预加载 `vectors.f32.bin`（`preloadVectors`）

---

## 环境变量与配置项

你的代码里环境变量主要分两类：

1. **服务相关**（Fastify 启动用）
2. **DB 参数相关**（open / init / 搜索策略用）

> 优先级（以 DB 参数为例）：
> **显式 opts** > **环境变量** > **默认值 / preset 推导**

### 1）服务端环境变量（`src/api/server.ts`）

| 环境变量                 | 作用                       |                    默认值 | 生效时机 | 备注                                                   |          |           |
| ------------------------ | -------------------------- | ------------------------: | -------- | ------------------------------------------------------ | -------- | --------- |
| `API_HOST`               | HTTP 服务绑定地址          |               `127.0.0.1` | 启动时   | 例如 `0.0.0.0` 用于容器/局域网访问                     |          |           |
| `API_PORT`               | HTTP 服务端口              |                    `3000` | 启动时   |                                                        |          |           |
| `MINIVECTOR_STORAGE_DIR` | DB 数据目录                |      `process.cwd()/data` | 启动时   | server.ts 里会传给 `MiniVectorDB.open({ storageDir })` |          |           |
| `MODEL_NAME`             | embedding 模型名           | `Xenova/all-MiniLM-L6-v2` | 启动时   | 同时影响 arch/dim 推导                                 |          |           |
| `MINIVECTOR_MODE`        | 档位 preset                |                `balanced` | 启动时   | `fast                                                  | balanced | accurate` |
| `HNSW_CAPACITY`          | capacity 上限              |               `1_200_000` | 启动时   | 容量不足会直接报错（见 FAQ）                           |          |           |
| `PRELOAD_VECTORS`        | 是否预加载 vectors.f32.bin |                       `0` | 启动时   | `1` 表示 preload（更快但更占内存）                     |          |           |

### 2）库环境变量（`src/index.ts` 内 resolveOpenConfig）

| 环境变量                 | 对应配置项         | 作用                |                          默认值 | 生效时机                   | 是否需要重建                                 |
| ------------------------ | ------------------ | ------------------- | ------------------------------: | -------------------------- | -------------------------------------------- |
| `MINIVECTOR_STORAGE_DIR` | `storageDir`       | 数据目录            |                        `./data` | open 时                    | 否                                           |
| `MODEL_NAME`             | `modelName`        | embedding 模型名    |       `Xenova/all-MiniLM-L6-v2` | open 时                    | **是（通常）**                               |
| `MINIVECTOR_MODE`        | `mode`             | 一键档位            |                      `balanced` | open 时                    | 取决于改了哪些参数（见下）                   |
| `VECTOR_DIM`             | `dim`              | 向量维度            | 从模型推导（text=384/clip=512） | open/init 时               | **是**                                       |
| `HNSW_CAPACITY`          | `capacity`         | internal_id 上限    |                     `1_200_000` | init/插入时                | 改大不需要重建，但现有文件要匹配你的使用方式 |
| `PRELOAD_VECTORS`        | `preloadVectors`   | 预加载 vectors      |                             `0` | load 时                    | 否                                           |
| `HNSW_M`                 | `m`                | HNSW M              |                          preset | init 时                    | **是**（dump 不可复用）                      |
| `HNSW_EF`                | `ef_construction`  | 建库 efConstruction |                          preset | init 时                    | **是**（dump 不可复用）                      |
| `BASE_EF_SEARCH`         | `baseEfSearch`     | 搜索 efSearch 基线  |                          preset | 查询时（动态 setEfSearch） | 否                                           |
| `RERANK_MULTIPLIER`      | `rerankMultiplier` | annK = topK \* mult |                          preset | 查询时                     | 否                                           |
| `MAX_ANN_K`              | `maxAnnK`          | annK 上限           |                          preset | 查询时                     | 否                                           |
| `HNSW_RESULTS_CAP`       | `resultsCap`       | WASM 结果缓冲上限   |                          preset | init / 查询时可能增长      | 否（但受 wasm MAX_EF 限制）                  |

#### “是否需要重建”的判断规则（实用版）

- **必须重建/旧 dump 不能再用**：`dim`、`m`、`ef_construction`、embedding 模型（通常意味着 dim/分布变化）
- **不需要重建（可运行时调）**：`baseEfSearch`、`rerankMultiplier`、`maxAnnK`、`resultsCap`、`preloadVectors`
- `capacity`：改大通常不要求重建索引，但你需要保证你的文件/元数据使用方式一致（更常见做法是：规划一个足够大容量，并长期不变）

---

## 档位与调参建议

### 三档 preset（你当前代码）

- `fast`
  - 建库更快，候选池更小，延迟低，但 recall/ndcg 更容易下降

- `balanced`
  - 推荐默认：速度与质量更均衡

- `accurate`
  - 目标是更高 recall，但要避免把候选池默认怼到上限导致 p95/p99 明显上升

### 搜索阶段自动策略（你的实现）

- `efSearch = max(baseEfSearch, topK*2)`
- `annK = min(topK*rerankMultiplier, maxAnnK, wasmMaxEf)`
- resultsCap 会自动扩容到 ≥ annK（指数翻倍）

#### 调参优先级（最常用）

- 想更准：优先加 `rerankMultiplier` 或 `baseEfSearch`（会增大候选池/搜索深度）
- 想更快：优先减 `rerankMultiplier`（候选池直接变小），其次减 `baseEfSearch`
- “建库更准”：加 `m`/`ef_construction`，但会显著影响建库时间和内存，并且需要重建

---

## FAQ / Troubleshooting

### 1）重建索引会不会损坏之前的数据？

不会“神秘损坏”，但要区分三种文件：

- `metadata.json`：映射与 metadata
- `vectors.f32.bin`：原始向量
- `dump.bin`：HNSW 图结构 + 量化向量（与配置强绑定）

如果你改了 `dim / m / ef_construction` 或更换 embedding 模型：

- **旧的 dump.bin 基本不可复用**（应删除/迁移后重建）
- `vectors.f32.bin` 理论上仍可用，但如果 dim/模型变了，那这些向量本身也不再匹配新索引语义（应重新生成）

实操建议：

- 改建库参数（m/ef）/维度/模型：**备份 data 目录** → 删除 `dump.bin` → 重新插入或从源数据重建
- 仅改搜索策略（baseEfSearch/rerankMultiplier 等）：无需重建，直接重启即可

---

### 2）启动后好像“加载了但查不到”/“查出来很怪”

常见原因：

- **插入与查询使用了不同的模型**（modelName 变了）
- **维度不一致**（VECTOR_DIM 与实际 embedding 输出不一致）
- 你插入的是向量，但又在查询时用文本（或相反）导致 embedding 来源不一致

建议检查：

- `/stats` 输出的 `model/dim/mode`
- 确保插入与查询都走同一套 embedder 或同一模型

---

### 3）报错：`Vector dimension mismatch`

你的代码会在 insert/search 时强校验维度。
解决方法：

- 如果你传入的是向量：确保长度等于 `dim`
- 如果你传入的是文本/图片：确保 embedder 输出维度与你配置一致
  - 推荐不要手动写 `VECTOR_DIM`，让它从模型推导或在 open 里显式传 `dim`

---

### 4）报错：`Database capacity exceeded / overflow`

这说明 internal_id 超过 `capacity` 上限（硬上限）。
解决方法：

- 重新 `MiniVectorDB.open({ capacity: 更大 })`
- 并且建议从一开始就规划足够大（例如 2x~4x 预估峰值），避免频繁迁移

注意：

- capacity 只是上限，不代表会预分配同等大小的磁盘文件（你的写入按需发生）
- 但随着数据增长，`vectors.f32.bin` 最终会变大：约 `items * dim * 4 bytes`

---

### 5）WASM 相关：`MAX_EF=4096` 是什么？为什么我的 maxAnnK 超过没用？

你的 WASM 内部存在 `MAX_EF=4096` 的硬限制，因此：

- 实际 `annK` 最终会被 `wasmMaxEf` clamp（`Math.min(annK, wasmMaxEf)`）
- `resultsCap` 也不应该超过 4096，否则只是“配置上写了但实际没意义”

建议：

- 文档中明确写：**maxAnnK / resultsCap 上限 4096**
- accurate 档默认就用 4096 足够（不用再写 10000）

---

### 6）查询 p95/p99 很高，怎么降？

最常见的原因是候选池太大（rerankMultiplier 太高 / baseEfSearch 太高）。
建议按顺序做：

1. 降低 `RERANK_MULTIPLIER`（最直接：候选池变小，rerank 计算和 IO 变少）
2. 降低 `BASE_EF_SEARCH`
3. 如果你启用了 `PRELOAD_VECTORS=1`，对磁盘 IO 敏感场景会更稳（但 RSS 更高）
4. 如果你用的是机械盘或网络盘，`PRELOAD_VECTORS` 对尾延迟改善更明显

---

### 7）为什么 `score` 越小越相似？不是应该越大越相似吗？

你的实现返回的是 **L2 distance squared**（平方欧氏距离）：

- **越小越相似**
- 这与很多“cosine similarity 越大越相似”的库不同

如果你希望返回 cosine similarity：

- 因为你已 normalize 向量，L2 与 cosine 存在单调关系（可转换/重标定）
- 但保持 L2^2 作为内部计算更简单、更快

---

### 8）开启 `PRELOAD_VECTORS=1` 会发生什么？

- 会在 `load()` 时把 `vectors.f32.bin` 全部读入内存缓存（Buffer）
- 查询时 rerank 直接从内存读，减少随机 IO，p95/p99 更稳定
- 代价是 RSS 上升（向量文件越大，占用越多）

适用建议：

- 数据量较大且磁盘 IO 不稳定：建议开启
- 内存紧张：保持关闭

---

### 9）WASM SIMD 加载失败 / release.wasm 无法加载

可能原因：

- Node 版本太旧
- 运行时不支持 WASM SIMD（某些受限环境）

处理建议：

- 升级 Node 到 18+/20+
- 提供一个非 SIMD 的构建产物或 fallback（文档提示）
- 在 README 中写清楚：不支持 SIMD 时如何构建/切换 wasm 产物

---

### 10）我修改了 `MINIVECTOR_MODE`，为什么效果不明显？

原因可能是：

- 你只改了 mode，但同时又通过环境变量覆盖了 preset（例如 `HNSW_M`、`HNSW_EF`、`RERANK_MULTIPLIER` 等）
- 或者你改的是建库参数，但没有重建索引（dump 还是旧的）

建议排查：

- 打印 `db.cfg`（你 benchmark 输出里已有 chosenParams）
- 确认最终生效的参数来自哪里（opts/env/preset）
- 改建库参数后删除旧 `dump.bin` 并重建

---

## 项目结构

- `assembly/`：AssemblyScript 实现的 HNSW + 距离计算 + allocator
- `src/core/wasm-bridge.ts`：WASM 调用桥接、resultsCap/efSearch 管理、dump IO
- `src/storage/meta-db.ts`：LokiJS 元数据（external_id ↔ internal_id + metadata）
- `src/embedder.ts`：本地 embedding（`@xenova/transformers`）
- `src/index.ts`：MiniVectorDB 核心逻辑（open/insert/search/save/load）
- `src/api/server.ts`：Fastify HTTP API

---

## License

MIT
