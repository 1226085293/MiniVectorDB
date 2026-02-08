# HTTP API

Server: `src/api/server.ts` (Fastify)

## Start

```bash
API_PORT=3000 API_HOST=127.0.0.1 npm start
```

## POST /insert

Body:

```json
{
	"id": "doc:123",
	"input": "text | vector[] | binary",
	"metadata": { "tag": "notes" }
}
```

Returns:

```json
{ "status": "ok" }
```

## POST /search

Body:

```json
{
	"query": "text | vector[] | binary",
	"topK": 10,
	"filter": { "tag": "notes" },
	"score": "l2 | cosine | similarity"
}
```

Returns:

```json
{ "results": [{ "id": "...", "score": 0.12, "metadata": {} }] }
```

## POST /searchMany

Body:

```json
{
	"queries": ["...", "..."],
	"topK": 10,
	"filter": { "tag": "notes" },
	"score": "l2 | cosine | similarity"
}
```

Returns:

```json
{ "results": [ [ ... ], [ ... ] ] }
```

## POST /remove

Body:

```json
{ "id": "doc:123" }
```

or

```json
{ "ids": ["doc:1", "doc:2"] }
```

Returns:

```json
{ "status": "ok", "removed": 1, "missing": 0, "alreadyDeleted": 0 }
```

## POST /updateMetadata

Body:

```json
{
	"id": "doc:123",
	"metadata": { "tag": "new" },
	"merge": true
}
```

Returns:

```json
{ "status": "ok" }
```

## POST /rebuild

Body:

```json
{
	"capacity": 500000,
	"persist": true,
	"compact": true
}
```

Returns:

```json
{ "status": "ok", "rebuilt": 12345, "capacity": 500000, "compact": true }
```

## POST /save

Returns `{ "status": "saved" }`

## GET /stats

Returns:

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

Graceful shutdown (mostly for tests/CI).

Returns `{ "status": "shutting down" }`
