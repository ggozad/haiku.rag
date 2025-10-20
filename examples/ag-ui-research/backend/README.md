# Haiku.rag Research Assistant Backend

FastAPI backend for the haiku.rag interactive research assistant, using Pydantic AI with AG-UI protocol support.

## Setup

```bash
uv sync
uv run python main.py
```

The server starts on `http://localhost:8000` and uses [haiku.rag configuration](https://ggozad.github.io/haiku.rag/configuration/).

## Endpoints

- `GET /health` - Health check
- `POST /agent` - AG-UI protocol endpoint
