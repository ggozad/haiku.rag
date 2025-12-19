# haiku.rag Docker Compose Example

Run haiku.rag with docling-serve for remote document processing, file monitoring, and MCP server.

## Architecture

This example demonstrates remote processing with two services:

- **docling-serve** - Document conversion and chunking service
- **haiku-rag** - MCP server and file monitoring (using slim image)

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

The docker-compose.yml mounts three volumes:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/data` | Persistent LanceDB database |
| `./docs` | `/docs` | Documents to monitor and index |
| `./haiku.rag.yaml` | `/app/haiku.rag.yaml` | Configuration file |

**Important:** The `haiku.rag.yaml` config file must exist before running `docker compose up`. Copy it from the example:

```bash
cp haiku.rag.yaml.example haiku.rag.yaml
```

The example config sets `monitor.directories: [/docs]` - this is the **container path**, not your host path. Documents placed in `./docs` on your host will appear at `/docs` inside the container.

## Usage

```bash
# List documents
docker compose exec haiku-rag haiku-rag list

# Search
docker compose exec haiku-rag haiku-rag search "your query"

# Ask questions
docker compose exec haiku-rag haiku-rag ask "What is haiku.rag?"
```

## Ports

- `5001` - docling-serve API (with UI enabled)
- `8001` - MCP server

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
