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
docker run -p 8001:8001 \
  -v $(pwd)/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v $(pwd)/data:/data \
  haiku-rag
```

The container will automatically use the mounted `haiku.rag.yaml` configuration file.

For API keys (OpenAI, Anthropic, etc.), pass them as environment variables:

```bash
docker run -p 8001:8001 \
  -v $(pwd)/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v $(pwd)/data:/data \
  -e OPENAI_API_KEY=your-key-here \
  haiku-rag
```

## Docker Compose

See `examples/docker/` for a complete setup example.
