# Embedding / 模型

MiniVectorDB 内置默认 embedder：基于 `@xenova/transformers` 的本地推理。

## 文本 embedding（默认）

默认模型：

- `Xenova/all-MiniLM-L6-v2`
- dim: 384

## CLIP（图文多模态）

若 modelName 包含 `clip`，会识别为 `clip` 架构：

- dim: 512
- 支持：
  - 图片输入（Buffer/Uint8Array 或本地路径/URL，取决于你的接入方式）
  - 文本输入

## 离线 / 缓存

open 选项：

- `modelCacheDir`：指定 transformers 缓存目录
- `localFilesOnly`：尽量避免联网（离线/镜像预热场景）

注意：

- `dim` 必须和 embedder 输出一致
- 同一 DB 不支持混用不同模型/维度
