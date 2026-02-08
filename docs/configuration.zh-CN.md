# 配置

配置解析优先级：

**open() 显式 opts** > **环境变量** > **默认值 / preset**

## open 选项（DBOpenOptions）

常用字段：

- `storageDir?: string`  
  数据目录（所有文件都在这里）。

- `collection?: string`  
  可选集合名/前缀：文件会变成 `${collection}.metadata.json` 等。

- `modelName?: string`  
  默认：`Xenova/all-MiniLM-L6-v2`

- `modelArchitecture?: "text" | "clip"`  
  不填会从 `modelName` 推断（包含 clip 关键字则为 clip）。

- `dim?: number`  
  默认按架构推断：text=384，clip=512。必须与 embedder 输出一致。

- `mode?: "fast" | "balanced" | "accurate"`  
  档位 preset（影响 ANN/搜索策略默认值）。

- `capacity?: number`  
  internal_id 最大值，默认 `1_200_000`。

- `preloadVectors?: boolean`  
  true 时会预加载 `vectors.f32.bin` 进内存以加速精排（更占 RAM）。

- `m?: number` / `ef_construction?: number`  
  HNSW 建库参数。修改通常意味着旧 dump 不可复用，需要重建。

- `embeddingCacheSize?: number`  
  文本 embedding 缓存条数（近似 LRU）。

- `deletedRebuildThreshold?: number`  
  自动 rebuild 阈值：deletedSinceRebuild / total >= 阈值（默认 0.2）。

- `autoRebuildOnLoad?: boolean`  
  dump 缺失/损坏时是否自动从 vectors+metadata 重建（默认 true）。

- `modelCacheDir?: string` / `localFilesOnly?: boolean`  
  xenova/transformers 的离线/缓存相关设置。

## 环境变量

`src/config.ts` 与 `src/api/server.ts` 支持的变量：

### 服务端

- `API_HOST`（默认 `127.0.0.1`）
- `API_PORT`（默认 `3000`）
- `MINIVECTOR_STORAGE_DIR`
- `MINIVECTOR_COLLECTION`
- `MODEL_NAME`
- `MINIVECTOR_MODE`
- `HNSW_CAPACITY`
- `PRELOAD_VECTORS`

### DB 参数

- `MINIVECTOR_STORAGE_DIR`
- `MINIVECTOR_COLLECTION`
- `MODEL_NAME`
- `MINIVECTOR_MODE`
- `VECTOR_DIM`
- `HNSW_CAPACITY`
- `PRELOAD_VECTORS`
- `HNSW_M`
- `HNSW_EF`
- `BASE_EF_SEARCH`
- `RERANK_MULTIPLIER`
- `MAX_ANN_K`
- `HNSW_RESULTS_CAP`
- `MINIVECTOR_DELETED_REBUILD_THRESHOLD`
- `MINIVECTOR_AUTO_REBUILD_ON_LOAD`
- `MINIVECTOR_MODEL_CACHE_DIR`
- `MINIVECTOR_LOCAL_FILES_ONLY`
- `EMBEDDING_CACHE_SIZE`

## Preset 档位

- `fast`：更低延迟，召回池更小
- `balanced`：默认
- `accurate`：更高召回，更多 CPU/IO
