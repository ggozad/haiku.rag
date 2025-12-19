# haiku.rag Docker Image

The full haiku.rag Docker image includes all features and extras (docling, voyageai, mxbai). You can build it locally using the provided Dockerfile.

## Building the Image

Build the full image with all features:

```bash
docker build -f docker/Dockerfile -t haiku-rag .
```

This creates an image with:
- All document processing capabilities (Docling)
- VoyageAI embeddings
- MixedBread AI reranking
- Full feature set

## Configuration

Create a configuration file `haiku.rag.yaml`:

```yaml
# haiku.rag.yaml
environment: production

embeddings:
  model:
    provider: ollama
    name: nomic-embed-text
    vector_dim: 768

qa:
  model:
    provider: ollama
    name: qwen3
```

See [Configuration docs](https://ggozad.github.io/haiku.rag/configuration/) for all available options.

## Running

Mount your config file and data directory:

```bash
docker run -p 8001:8001 \
  -v /path/to/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v /path/to/data:/data \
  haiku-rag
```

To enable file monitoring, also mount a documents directory:

```bash
docker run -p 8001:8001 \
  -v /path/to/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v /path/to/data:/data \
  -v /path/to/docs:/docs \
  haiku-rag haiku-rag serve --mcp --monitor
```

Your `haiku.rag.yaml` must reference the **container path** for monitoring:

```yaml
monitor:
  directories:
    - /docs  # Container path, not host path
```

For API keys (OpenAI, Anthropic, etc.), pass them as environment variables:

```bash
docker run -p 8001:8001 \
  -v /path/to/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v /path/to/data:/data \
  -e OPENAI_API_KEY=your-key-here \
  haiku-rag
```

## Docker Compose

See `examples/docker/` for a complete setup example.
