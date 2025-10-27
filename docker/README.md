# haiku.rag Docker Image

Pre-built images are available at `ghcr.io/ggozad/haiku.rag` with all extras (voyageai, mxbai, a2a).

## Using Pre-built Image

```bash
docker pull ghcr.io/ggozad/haiku.rag:latest
```

## Configuration

Create a configuration file `haiku.rag.yaml`:

```yaml
# haiku.rag.yaml
environment: production

embeddings:
  provider: ollama
  model: nomic-embed-text
  vector_dim: 768

qa:
  provider: ollama
  model: qwen3
```

See [Configuration docs](https://ggozad.github.io/haiku.rag/configuration/) for all available options.

## Running

Mount your config file and data directory:

```bash
docker run -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v $(pwd)/data:/data \
  ghcr.io/ggozad/haiku.rag:latest
```

The container will automatically use the mounted `haiku.rag.yaml` configuration file.

For API keys (OpenAI, Anthropic, etc.), pass them as environment variables:

```bash
docker run -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v $(pwd)/data:/data \
  -e OPENAI_API_KEY=your-key-here \
  ghcr.io/ggozad/haiku.rag:latest
```

## Building Locally

```bash
docker build -f docker/Dockerfile -t haiku-rag .
```

## Docker Compose

See `examples/docker/` for a complete setup example.
