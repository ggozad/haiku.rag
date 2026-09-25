# haiku.rag Docker Compose Example

Run haiku.rag with docling-serve for remote document processing, continuous ingestion via `haiku-ingester`, and a read-only MCP server.

## Architecture

haiku.rag allows one writing process per database, with any number of
readers, so the example runs the ingester and the MCP server as **two
separate containers** sharing the same data volume:

- **docling-serve-1** / **docling-serve-2**: two replicas of the
  conversion and chunking service. The ingester round-robins jobs across
  them, so conversions overlap and one restarting does not stall ingest.
  For more replicas, duplicate the service block and add its URL to
  `providers.docling_serve.base_url`.
- **haiku-ingester**: the one writer. Watches `/docs`, ingests new and
  changed files, queues retries, exposes the control plane on port 8765.
- **haiku-rag**: read-only MCP server on port 8001.

Both haiku containers run the published slim image with the same config
file. Compose overrides the image's default command to give each its role.
The slim image has no Docling and converts through docling-serve.

## Quick Start

```bash
# Create required directories
mkdir -p data docs

# Create config file from example (required)
cp haiku.rag.yaml.example haiku.rag.yaml

# Start services (pulls ghcr.io/ggozad/haiku.rag-slim:latest)
docker compose up -d
```

Place documents in `docs/` for automatic indexing.

### Building locally for development

The example pulls the published image. To run a local build of
`haiku.rag-slim` instead, add a `docker-compose.override.yml` next to
`docker-compose.yml`, which Compose loads automatically:

```yaml
services:
  haiku-ingester:
    build:
      context: ../..
      dockerfile: docker/Dockerfile.slim
  haiku-rag:
    build:
      context: ../..
      dockerfile: docker/Dockerfile.slim
```

Then:

```bash
docker compose build         # builds & tags as ghcr.io/ggozad/haiku.rag-slim:latest
docker compose up -d         # uses the local image
docker compose pull          # back to the published image when done
```

## Volume Mounts

| Host Path | Container Path | Mounted on | Purpose |
|-----------|----------------|------------|---------|
| `./data` | `/data` | both haiku containers | Persistent LanceDB + ingester queue |
| `./docs` | `/docs` | `haiku-ingester` only | Documents to ingest (watched by the FS source) |
| `./haiku.rag.yaml` | `/app/haiku.rag.yaml` | both haiku containers | Configuration file |

`haiku.rag.yaml` must exist before `docker compose up`, or Docker creates a directory in its place. The example config sets `ingester.sources[0].root: /docs` - this is the **container path**, not your host path. Documents placed in `./docs` on your host will appear at `/docs` inside the container.

## Usage

Files dropped into `./docs/` on the host are picked up by the ingester, through
filesystem events and a periodic sweep.

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
source .env   # INGESTER_TOKEN
curl http://localhost:8765/health
curl -H "Authorization: Bearer $INGESTER_TOKEN" 'http://localhost:8765/jobs?status=queued'
curl -H "Authorization: Bearer $INGESTER_TOKEN" http://localhost:8765/dlq
```

## Ports

- `5001` - docling-serve replica 1 API (with UI enabled, debug only)
- `5002` - docling-serve replica 2 API (the container listens on 5001)
- `8001` - MCP server (read-only)
- `8765` - ingester control plane (`/health`, `/jobs`, `/sources`, `/dlq`)

docling-serve and the MCP server are published on `127.0.0.1` only. The MCP server has no authentication: anyone who can reach port 8001 can search and read every document and run `execute_code`. Put an authenticating proxy in front of it before publishing it beyond the host.

## Configuration

The setup uses `haiku.rag-slim` image configured to use docling-serve for document processing:

```yaml
processing:
  converter: docling-serve
  chunker: docling-serve

providers:
  docling_serve:
    base_url:
      - http://docling-serve-1:5001
      - http://docling-serve-2:5001
```

Edit `haiku.rag.yaml` to configure providers, embeddings, and other settings. See the [Configuration documentation](https://ggozad.github.io/haiku.rag/configuration/) for all options.

For API keys (OpenAI, Anthropic, etc.), set them as environment variables:

```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
docker compose up -d
```

The ingester container binds the control plane to `0.0.0.0` so the host
port-mapping works. The example config requires a bearer token via
`INGESTER_TOKEN`. Both haiku containers load that config and fail to start
without it, so set it in `.env` (gitignored) alongside the API keys before
bringing the stack up:

```bash
echo "INGESTER_TOKEN=$(openssl rand -hex 32)" >> .env
```

### Using a database server for the queue

By default the queue is a SQLite file on the `./data` volume. To run it on
Postgres instead, point `ingester.queue.dburi` at the server in
`haiku.rag.yaml`:

```yaml
ingester:
  queue:
    dburi: postgresql+asyncpg://haiku:secret@postgres:5432/haiku_rag
```

Add a Postgres service and wire the ingester to it with a
`docker-compose.override.yml` (auto-loaded by Compose):

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=haiku
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=haiku_rag
    volumes:
      - ./pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "haiku", "-d", "haiku_rag"]
      interval: 5s
      timeout: 3s
      retries: 12
    restart: unless-stopped

  haiku-ingester:
    depends_on:
      postgres:
        condition: service_healthy
```

Workers claim jobs with `FOR UPDATE SKIP LOCKED`, so several ingesters can
share one Postgres queue. Each still needs a database of its own: the
one-writer rule is haiku.rag's and holds on every store, LanceDB Cloud
included.

## Documentation

- [Remote Processing](https://ggozad.github.io/haiku.rag/remote-processing/)
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/)
- [CLI Commands](https://ggozad.github.io/haiku.rag/cli/)
- [MCP Server](https://ggozad.github.io/haiku.rag/mcp/)
