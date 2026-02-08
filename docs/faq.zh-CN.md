# FAQ

## 1）WASM SIMD 加载失败

如果运行环境不支持 WASM SIMD，开启 SIMD 的 `release.wasm` 可能无法加载。

解决思路：

- 使用支持 WASM SIMD 的 Node/运行环境（通常 Node 18/20 支持）
- 维护一个不启用 SIMD 的构建目标
- 确保部署环境开启 SIMD

## 2）“Vector dimension mismatch”

同一个 DB 内所有向量必须与配置 `dim` 一致。
更换模型/架构后，建议使用新的 storageDir 或 rebuild。

## 3）删除后磁盘不降

删除是软删除。要回收空间：

- `db.rebuild({ compact: true })`

## 4）如何扩容 capacity

capacity 是 internal_id 上限，超过会报错。
解决：

- 用更大 capacity 重建
  - `db.rebuild({ capacity: old*2, compact: false })`
  - 或 `compact: true`（会重写文件并压缩 ID）
