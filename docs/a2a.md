# Agent-to-Agent (A2A) Protocol

The A2A server exposes `haiku.rag` as a conversational agent using the Agent-to-Agent protocol. Unlike the MCP server which provides stateless tools, the A2A agent maintains conversation history and context across multiple turns.

## Features

- **Conversational Context**: Maintains full conversation history including tool calls and results
- **Multi-turn Dialogue**: Supports follow-up questions with pronoun resolution ("he", "it", "that document")
- **Intelligent Search**: Performs single or multiple searches depending on question complexity
- **Source Citations**: Always includes sources with both titles and URIs
- **Full Document Retrieval**: Can fetch complete documents on request
- **Document Discovery**: Lists available documents to help users explore the knowledge base

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
