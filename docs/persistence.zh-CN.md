# 持久化与文件结构

所有数据文件都在 `storageDir` 下（可选 `collection` 前缀）。

## 文件列表

- `metadata.json`
  - LokiJS 数据库
  - 保存：`external_id`、`internal_id`、`metadata`、软删除字段

- `vectors.f32.bin`
  - 连续的 float32 向量二进制
  - 偏移：`internal_id * dim * 4`

- `dump.bin`
  - WASM ANN 快照（图结构 + 向量 + 配置头）
  - 加载时会校验：dim/m/mmax0/ef_construction/max_layers 等必须一致

- `state.json`
  - 记录快照时间与计数、capacity 等信息

- `ann.oplog`
  - 操作日志：
    - `U <internal_id>`：upsert
    - `D <internal_id>`：delete 标记（目前删除主要在 metadata 层过滤）
  - dump 加载成功后会回放，追平最近写入

## Save / Load 流程

- `open()`：完成 init 并尝试 `load()`
- `load()`：
  - 优先尝试加载 `dump.bin`
  - dump 缺失/损坏且 `autoRebuildOnLoad=true`：从 vectors+metadata 重建 ANN
  - 回放 `ann.oplog` 追平最近写入

- `save()`：
  - flush metadata
  - sync 向量文件
  - dump 写 tmp 再 atomic rename
  - 快照完成后会截断 `ann.oplog`

## 重建与压缩

- `rebuild({ compact: false })`：
  - 重新 init WASM index
  - 按 internal_id 扫 vectors，插入存活项
  - 不重写文件、不重排 ID

- `rebuild({ compact: true })`（默认）：
  - 重写 `vectors.f32.bin`：仅保留存活项
  - 重写 `metadata.json`：internal_id 压缩为连续
  - 基于新文件重建 WASM ANN
