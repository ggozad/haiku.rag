# haiku-rag-a2a

A2A (Agent-to-Agent) protocol server for haiku.rag. This package provides a conversational agent interface that maintains conversation history and context across multiple turns.

## Features

- **Conversational Context**: Maintains full conversation history including tool calls and results
- **Multi-turn Dialogue**: Supports follow-up questions with pronoun resolution ("he", "it", "that document")
- **Intelligent Search**: Performs single or multiple searches depending on question complexity
- **Source Citations**: Always includes sources with both titles and URIs
- **Full Document Retrieval**: Can fetch complete documents on request
- **Multiple Skills**: Exposes three distinct skills with appropriate artifacts:
  - `document-qa`: Conversational question answering (default)
  - `document-search`: Semantic search with structured results
  - `document-retrieve`: Fetch complete documents by URI

## Installation

This package is not published to PyPI. Install it locally from the haiku.rag repository:

```bash
cd examples/a2a-server
uv sync
```

This will install the package and all its dependencies, including `haiku.rag`.

## Quick Start

### Starting the A2A Server

```bash
# Start server with default database location (uses the same default as haiku-rag)
uv run haiku-rag-a2a serve

# Or specify a custom database path
uv run haiku-rag-a2a serve --db /path/to/database

# Start on custom host/port
uv run haiku-rag-a2a serve --host 0.0.0.0 --port 8080
```

By default, the server uses the same database location as `haiku-rag`:
- Linux: `~/.local/share/haiku.rag`
- macOS: `~/Library/Application Support/haiku.rag`
- Windows: `C:/Users/<USER>/AppData/Roaming/haiku.rag`

### Interactive Client

Test and interact with the A2A server using the built-in interactive client:

```bash
# Connect to local server
uv run haiku-rag-a2a client

# Connect to remote server
uv run haiku-rag-a2a client --url https://example.com:8000
```

The interactive client provides:
- Rich markdown rendering of agent responses
- Conversation context across multiple turns
- Agent card discovery and display
- Compact artifact summaries

## Python Usage

```python
from pathlib import Path
from haiku_rag_a2a.a2a import create_a2a_app
import uvicorn

# Create A2A app
app = create_a2a_app(Path("/path/to/database"))

# Run with uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Security Examples

The `security_examples/` directory contains examples for securing the A2A server:

- `apikey_example.py` - Simple API key authentication
- `oauth2_github.py` - GitHub Personal Access Token authentication
- `oauth2_example.py` - Full OAuth2 with JWT verification

## Architecture

The A2A agent uses:

- **FastA2A**: Python framework implementing the A2A protocol
- **Pydantic AI**: Agent framework with tool support
- **In-Memory Storage**: Context and message history storage (persists during server lifetime)
- **Conversation State**: Full pydantic-ai message history serialized in A2A context

## Configuration

The server uses the same configuration as haiku.rag. You can specify a config file:

```bash
uv run haiku-rag-a2a serve --db /path/to/database --config haiku.rag.yaml
```

You can also control the maximum number of conversation contexts via the `--max-contexts` parameter (defaults to 1000).

## Documentation

See [a2a.md](./a2a.md) for detailed documentation including:
- API examples
- Security configuration
- Docker deployment
- Artifact specification

## License

MIT
