# 调参指南

MiniVectorDB：

- WASM HNSW 类 ANN 做召回
- 可选用 float32 原始向量精排提升质量

## 三档 preset

- `fast`
  - `m` 较小、`ef_construction` 适中
  - `baseEfSearch` / `rerankMultiplier` 更小
  - 延迟更低，召回可能下降

- `balanced`（默认）
  - 速度/质量折中

- `accurate`
  - 更大的 `m`，更大的 `ef_construction`
  - 候选池更大
  - 召回更稳，但 CPU/IO 更高

## 关键参数

### 建库期（修改通常导致旧 dump 不可复用）

- `dim`
- `m`
- `ef_construction`
- embedding 模型/架构变化

### 查询期（可运行时调整，不影响文件布局）

- `baseEfSearch`：efSearch 基线（代码会保证 efSearch >= topK\*2）
- `rerankMultiplier`：候选数 = topK \* multiplier（再被 maxAnnK 限制）
- `maxAnnK`：候选池上限
- `resultsCap`：WASM 结果缓冲上限（受 MAX_EF=4096 限制）
- `preloadVectors`：用内存换 IO

## 实用建议

- 先用 `balanced` 起步。
- 召回不稳：优先加 `baseEfSearch` 与 `rerankMultiplier`。
- p95 IO 抖动：降低 `rerankMultiplier` 或开启 `preloadVectors`（前提是内存允许）。
- 数据量大：提前规划 capacity；定期 compact rebuild 可回收空间并改善局部性。
