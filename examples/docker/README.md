# haiku.rag Docker Compose Example

Run haiku.rag with docling-serve for remote document processing, continuous ingestion via `haiku-ingester`, and a read-only MCP server.

## Architecture

LanceDB allows exactly one writer + N readers per database URI, so the
example runs the ingester and the MCP server as **two separate containers**
sharing the same data volume:

- **docling-serve** - Document conversion and chunking service
- **haiku-ingester** - Long-lived writer. Watches `/docs`, ingests new and
  changed files, queues retries, exposes the control plane on port 8765.
- **haiku-rag** - Read-only MCP server on port 8001 for AI assistant
  integration. Cannot write to the database — the ingester owns writes.

Both haiku.* services share the same slim image (built once) and the same
config file; docker-compose overrides the image's default command to give
each container its role.

This setup showcases the minimal haiku.rag-slim image combined with external document processing, ideal for production deployments.

## Quick Start

```bash
# Create required directories
mkdir -p data docs

# Create config file from example (required)
cp haiku.rag.yaml.example haiku.rag.yaml

# Start services
docker compose up -d
```

Place documents in `docs/` for automatic indexing.

## Volume Mounts

| Host Path | Container Path | Mounted on | Purpose |
|-----------|----------------|------------|---------|
| `./data` | `/data` | both haiku containers | Persistent LanceDB + ingester queue |
| `./docs` | `/docs` | `haiku-ingester` only | Documents to ingest (watched by the FS source) |
| `./haiku.rag.yaml` | `/app/haiku.rag.yaml` | both haiku containers | Configuration file |

**Important:** The `haiku.rag.yaml` config file must exist before running `docker compose up`. Copy it from the example:

```bash
cp haiku.rag.yaml.example haiku.rag.yaml
```

The example config sets `ingester.sources[0].root: /docs` - this is the **container path**, not your host path. Documents placed in `./docs` on your host will appear at `/docs` inside the container.

## Usage

Add documents by dropping files into `./docs/` on the host — the ingester
picks them up automatically (watchfiles + periodic sweep).

The `haiku-rag` container runs in read-only mode, so use it for queries:

```bash
# List documents
docker compose exec haiku-rag haiku-rag list

# Search
docker compose exec haiku-rag haiku-rag search "your query"

# Ask questions
docker compose exec haiku-rag haiku-rag ask "What is haiku.rag?"
```

Check ingester progress via its control plane:

```bash
curl http://localhost:8765/health
curl http://localhost:8765/jobs?status=queued
curl http://localhost:8765/dlq
```

## Ports

- `5001` - docling-serve API (with UI enabled)
- `8001` - MCP server (read-only)
- `8765` - ingester control plane (`/health`, `/jobs`, `/sources`, `/dlq`)

## Configuration

The setup uses `haiku.rag-slim` image configured to use docling-serve for document processing:

```yaml
processing:
  converter: docling-serve
  chunker: docling-serve

providers:
  docling_serve:
    base_url: http://docling-serve:5001
```

Edit `haiku.rag.yaml` to configure providers, embeddings, and other settings. See the [Configuration documentation](https://ggozad.github.io/haiku.rag/configuration/) for all options.

For API keys (OpenAI, Anthropic, etc.), set them as environment variables:

```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
docker compose up -d
```

## Documentation

- [Remote Processing](https://ggozad.github.io/haiku.rag/remote-processing/)
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/)
- [CLI Commands](https://ggozad.github.io/haiku.rag/cli/)
- [MCP Server](https://ggozad.github.io/haiku.rag/mcp/)
