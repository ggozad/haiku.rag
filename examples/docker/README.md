# haiku.rag Docker Compose Example

Run haiku.rag with file monitoring and MCP server.

## Quick Start

```bash
mkdir -p data docs
cp haiku.rag.yaml.example haiku.rag.yaml  # Edit as needed
docker compose up -d
```

Place documents in `docs/` for automatic indexing.

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

- `8001` - MCP server

## Configuration

Edit `haiku.rag.yaml` to configure providers, embeddings, and other settings. See the [Configuration documentation](https://ggozad.github.io/haiku.rag/configuration/) for all options.

Default setup uses Ollama on the host (`host.docker.internal:11434`).

For API keys (OpenAI, Anthropic, etc.), set them as environment variables:

```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
docker compose up -d
```

## Documentation

- [Configuration](https://ggozad.github.io/haiku.rag/configuration/)
- [CLI Commands](https://ggozad.github.io/haiku.rag/cli/)
- [MCP Server](https://ggozad.github.io/haiku.rag/mcp/)
