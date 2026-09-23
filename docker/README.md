# haiku.rag Docker Image

The full haiku.rag Docker image contains the `haiku.rag` package (Docling, VoyageAI and Cohere embedders, every reranker, the terminal UI) and the `ingester` extra. It is not published. Build it locally with the provided Dockerfile.

## Building the Image

Build the full image with all features:

```bash
docker build -f docker/Dockerfile -t haiku-rag .
```

Its default command runs the read-only MCP server on port 8001, bound to `0.0.0.0`.

## Configuration

Create a configuration file `haiku.rag.yaml`:

```yaml
# haiku.rag.yaml
storage:
  data_dir: /data   # the mounted volume; without it the database stays inside the container

embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560

qa:
  model:
    provider: ollama
    name: qwen3.8

providers:
  ollama:
    base_url: http://host.docker.internal:11434
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
  api:
    host: 0.0.0.0             # reachable through the -p 8765:8765 mapping
    auth_token: ${INGESTER_TOKEN}
  queue:
    path: /data/ingester.db   # persist queue in the data volume
  sources:
    - type: fs
      id: docs
      root: /docs             # container path, not host path
      delete_orphans: true
```

The MCP server always opens the database read-only, so it can run beside
the ingester, which is the database's one writer. haiku.rag allows one
writing process per database. Pass `-e INGESTER_TOKEN=...` to both
containers, since both load the same configuration. See
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
