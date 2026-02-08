# Embedding models

MiniVectorDB ships with a default local embedder based on `@xenova/transformers`.

## Text embedding (default)

Default model:

- `Xenova/all-MiniLM-L6-v2`
- dim: 384

## CLIP (text+image)

If model name contains `clip`, MiniVectorDB treats it as `clip` architecture:

- dim: 512
- supports:
  - image inputs (Buffer/Uint8Array or local path / URL depending on your usage)
  - text inputs

## Offline / caching

Open options:

- `modelCacheDir`: set transformers cache directory
- `localFilesOnly`: best-effort offline mode (avoid network)

Notes:

- Ensure `dim` matches the embedder output.
- Mixing models/dims in the same DB is not supported.
