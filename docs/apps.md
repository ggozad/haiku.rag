# Web application

A browser-based reference implementation of conversational RAG, built on a Starlette backend with pydantic-ai's `AGUIAdapter` and a Next.js / CopilotKit frontend. It lives in the `app/` directory of the haiku.rag repository.

This is a starting point for your own deployments, not the canonical haiku.rag UX. For the day-to-day terminal experience see [Chat](chat.md).

## Features

- Streaming chat with real-time tool execution visibility.
- Expandable citations with source documents, pages, and headings.
- Visual grounding to view chunk source locations in documents.
- Document filter to restrict searches to selected documents.
- Session state view for inspecting citations and search results.

## Quick start

```bash
cd app
docker compose -f docker-compose.dev.yml up -d --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8001`

## Architecture

- **Backend**: Starlette server with pydantic-ai `AGUIAdapter`.
- **Frontend**: Next.js with CopilotKit.
- **Protocol**: AG-UI for streaming chat.

## Configuration

Create a `.env` file in the `app/` directory:

```bash
# API Keys (at least one required)
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key

# Database path
DB_PATH=/path/to/your/haiku.rag.lancedb

# Optional: Ollama base URL (if using local models)
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Logfire for observability
LOGFIRE_TOKEN=your-logfire-token
```

For full configuration, mount a `haiku.rag.yaml` file:

```yaml
# app/haiku.rag.yaml
qa:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/stream` | POST | AG-UI chat streaming |
| `/api/documents` | GET | List all documents |
| `/api/info` | GET | Database statistics |
| `/api/visualize/{chunk_id}` | GET | Visual grounding images (base64) |
| `/health` | GET | Health check |

## Development

The backend reloads automatically on file changes. For frontend changes:

```bash
docker compose -f docker-compose.dev.yml up -d --build frontend
```

If `LOGFIRE_TOKEN` is set, LLM calls are traced and available in the Logfire dashboard.
