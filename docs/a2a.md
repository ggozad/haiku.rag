# Agent-to-Agent (A2A) Protocol

The A2A server exposes `haiku.rag` as a conversational agent using the Agent-to-Agent protocol. Unlike the MCP server which provides stateless tools, the A2A agent maintains conversation history and context across multiple turns.

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

## Starting A2A Server

```bash
haiku-rag serve --a2a
```

Server options:
- `--a2a-host` - Host to bind to (default: 127.0.0.1)
- `--a2a-port` - Port to bind to (default: 8000)

Example:
```bash
haiku-rag serve --a2a --a2a-host 0.0.0.0 --a2a-port 8080
```

## Requirements

A2A support requires the `a2a` extra:

```bash
uv pip install 'haiku.rag[a2a]'
```

## Python Usage

```python
from pathlib import Path
from haiku.rag.a2a import create_a2a_app
import uvicorn

# Create A2A app
app = create_a2a_app(Path("database.lancedb"))

# Run with uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000)
```

This installs the `fasta2a` package and its dependencies.

## Architecture

The A2A agent uses:

- **FastA2A**: Python framework implementing the A2A protocol
- **Pydantic AI**: Agent framework with tool support
- **In-Memory Storage**: Context and message history storage (persists during server lifetime)
- **Conversation State**: Full pydantic-ai message history serialized in A2A context

### Message History

The agent stores the complete conversation state including:

- User prompts
- Agent responses
- Tool calls and their arguments
- Tool return values

This enables the agent to:

- Reference previous searches
- Understand pronouns and context
- Maintain coherent multi-turn conversations

### Context Management

Each conversation is identified by a `context_id`. All messages within the same context share conversation history. This allows the agent to:

- Remember what was discussed
- Track which documents were already found
- Provide contextual follow-up answers

### Skills

The agent exposes three skills:

- **document-qa** (default): Conversational question answering including follow-ups and multi-turn dialogue
- **document-search**: Direct semantic search returning formatted results
- **document-retrieve**: Fetch complete document content by URI

### Artifacts

All operations create artifacts for traceability:

- **search_results**: Created for each `search_documents` tool call

  - Contains query and array of SearchResult objects (content, score, document_title, document_uri)

- **document**: Created for each `get_full_document` tool call

  - Contains complete document text

- **qa_result**: Created for all document-qa operations

  - Contains question, answer, and skill identifier
  - Always created for Q&A, even when answering from conversation history without tools

### Memory Management

To prevent memory growth, the server uses LRU (Least Recently Used) eviction:

- Maximum 1000 contexts kept in memory (configurable via `A2A_MAX_CONTEXTS`)
- When limit exceeded, least recently used contexts are automatically evicted

Configure via environment variable:
```bash
export A2A_MAX_CONTEXTS=1000
```

## Security

By default, the A2A agent runs without authentication. For production deployments, you should add authentication.

### Adding Authentication

The `create_a2a_app()` function accepts optional security parameters that declare authentication requirements in the agent card:

```python
from haiku.rag.a2a import create_a2a_app

app = create_a2a_app(
    db_path,
    security_schemes={
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication",
        }
    },
    security=[{"apiKeyAuth": []}],
)
```

This populates the agent card at `/.well-known/agent-card.json` so other agents can discover your authentication requirements.

### Security Examples

Three working examples are provided in `examples/a2a-security/`:

1. **API Key** (`apikey_example.py`) - Simple header-based authentication
2. **OAuth2 GitHub** (`oauth2_github.py`) - GitHub Personal Access Token authentication
3. **OAuth2 Enterprise** (`oauth2_example.py`) - Full OAuth2 with JWT verification

Each example shows:

- How to declare security in the agent card
- How to implement authentication middleware
- How to verify credentials
