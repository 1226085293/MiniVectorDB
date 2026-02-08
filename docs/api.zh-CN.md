# HTTP API

服务端：`src/api/server.ts`（Fastify）

## 启动

```bash
API_PORT=3000 API_HOST=127.0.0.1 npm start
```

## POST /insert

请求体：

```json
{
	"id": "doc:123",
	"input": "文本 | 向量数组 | 二进制",
	"metadata": { "tag": "notes" }
}
```

返回：

```json
{ "status": "ok" }
```

## POST /search

请求体：

```json
{
	"query": "文本 | 向量数组 | 二进制",
	"topK": 10,
	"filter": { "tag": "notes" },
	"score": "l2 | cosine | similarity"
}
```

返回：

```json
{ "results": [{ "id": "...", "score": 0.12, "metadata": {} }] }
```

## POST /searchMany

请求体：

```json
{
	"queries": ["...", "..."],
	"topK": 10,
	"filter": { "tag": "notes" },
	"score": "l2 | cosine | similarity"
}
```

返回：

```json
{ "results": [ [ ... ], [ ... ] ] }
```

## POST /remove

请求体：

```json
{ "id": "doc:123" }
```

或：

```json
{ "ids": ["doc:1", "doc:2"] }
```

返回：

```json
{ "status": "ok", "removed": 1, "missing": 0, "alreadyDeleted": 0 }
```

## POST /updateMetadata

请求体：

```json
{
	"id": "doc:123",
	"metadata": { "tag": "new" },
	"merge": true
}
```

返回：

```json
{ "status": "ok" }
```

## POST /rebuild

请求体：

```json
{
	"capacity": 500000,
	"persist": true,
	"compact": true
}
```

返回：

```json
{ "status": "ok", "rebuilt": 12345, "capacity": 500000, "compact": true }
```

## POST /save

返回 `{ "status": "saved" }`

## GET /stats

返回：

```json
{
	"mode": "balanced",
	"collection": "optional",
	"model": "Xenova/all-MiniLM-L6-v2",
	"dim": 384,
	"items": 100,
	"deletedCount": 3,
	"activeCount": 97,
	"storageDir": "...",
	"capacity": 1200000,
	"preloadVectors": false,
	"wasmMaxEf": 4096
}
```

## POST /shutdown

优雅退出（主要用于测试/CI）。

返回 `{ "status": "shutting down" }`
