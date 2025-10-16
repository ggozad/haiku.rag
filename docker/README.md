# haiku.rag Docker Image

Dockerfile for building haiku.rag with all extras (voyageai, mxbai, a2a).

## Building

```bash
docker build -f docker/Dockerfile -t haiku-rag .
```

## Running

```bash
docker run -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/data:/data \
  -e EMBEDDINGS_PROVIDER=ollama \
  -e EMBEDDINGS_MODEL=nomic-embed-text \
  -e QA_PROVIDER=ollama \
  -e QA_MODEL=qwen3\
  haiku-rag
```

Note: The environment variables above override the defaults. See [Configuration docs](https://ggozad.github.io/haiku.rag/configuration/) for all options.

## Docker Compose

See `examples/docker/` for a complete setup example.
