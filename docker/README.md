# haiku.rag Docker Image

Pre-built images are available at `ghcr.io/ggozad/haiku.rag` with all extras (voyageai, mxbai, a2a).

## Using Pre-built Image

```bash
docker pull ghcr.io/ggozad/haiku.rag:latest
```

## Running

```bash
docker run -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/data:/data \
  -e EMBEDDINGS_PROVIDER=ollama \
  -e EMBEDDINGS_MODEL=nomic-embed-text \
  -e QA_PROVIDER=ollama \
  -e QA_MODEL=qwen3\
  ghcr.io/ggozad/haiku.rag:latest
```

## Building Locally

```bash
docker build -f docker/Dockerfile -t haiku-rag .
```

Note: The environment variables above override the defaults. See [Configuration docs](https://ggozad.github.io/haiku.rag/configuration/) for all options.

## Docker Compose

See `examples/docker/` for a complete setup example.
