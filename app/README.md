# haiku.rag Chat App

A conversational RAG interface built with [CopilotKit](https://copilotkit.ai/) and [pydantic-ai](https://github.com/pydantic/pydantic-ai)'s AG-UI protocol.

## Prerequisites

- Docker and Docker Compose
- A haiku.rag database (created via the `haiku-rag` CLI)
- An LLM API key (Anthropic, OpenAI, or local Ollama)

## Quick Start

1. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database path
   ```

2. **Configure the LLM and embedding models:**

   ```bash
   cp haiku.rag.yaml.example haiku.rag.yaml
   # Edit haiku.rag.yaml to configure your models
   ```

3. **Start the app:**

   ```bash
   docker compose up -d
   ```

4. **Open the chat interface:** http://localhost:3000

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DB_PATH` | Path to your haiku.rag LanceDB database | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key | One LLM key required |
| `OPENAI_API_KEY` | OpenAI API key | One LLM key required |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://host.docker.internal:11434`) | For local models |
| `LOGFIRE_TOKEN` | Pydantic Logfire token for debugging | No |

### haiku.rag.yaml

Configure the chat agent's LLM, embeddings, and search settings:

```yaml
qa:
  model:
    provider: anthropic  # or openai, ollama
    name: claude-sonnet-4-20250514

embeddings:
  model:
    provider: ollama
    name: nomic-embed-text

search:
  limit: 10
  context_radius: 1
```

See `haiku.rag.yaml.example` for all options.

## Development

For local development with hot reloading:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

- Backend code changes reload automatically
- Frontend available at http://localhost:3000
- Backend API at http://localhost:8001

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │────▶│   haiku.rag     │
│  (CopilotKit)   │     │  (pydantic-ai)  │     │   (LanceDB)     │
│  localhost:3000 │     │  localhost:8001 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/stream` | POST | AG-UI chat streaming |
| `/api/documents` | GET | List documents in database |
| `/api/info` | GET | Database statistics |
| `/api/visualize/{chunk_id}` | GET | Visual grounding for chunks |
| `/health` | GET | Health check |

## Chat Capabilities

The chat agent can:

- **Search** your documents with hybrid vector + full-text search
- **Answer questions** with citations from your knowledge base
- **Filter by document** when you ask about specific files
- **Show visual grounding** for PDF/image sources
