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

For continuous ingestion of a watched directory, run `haiku-ingester` in a
separate container against the same data volume:

```bash
docker run \
  -v /path/to/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v /path/to/data:/data \
  -v /path/to/docs:/docs \
  -p 8765:8765 \
  haiku-rag haiku-ingester --config /app/haiku.rag.yaml serve
```

Configure the watched directory in `haiku.rag.yaml` using the **container
path**:

```yaml
ingester:
  queue:
    path: /data/ingester.db   # persist queue in the data volume
  sources:
    - type: fs
      id: docs
      root: /docs             # container path, not host path
      delete_orphans: true
```

The MCP server running in the first container must be started with
`--read-only` when an ingester is writing to the same database — LanceDB
allows one writer and N readers per URI. See
`examples/docker/docker-compose.yml` for a working two-service setup.

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
