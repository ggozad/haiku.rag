# Haiku.rag Research Assistant Backend

Starlette backend for the haiku.rag interactive research assistant, using the research graph with AG-UI protocol support.

## Setup

```bash
uv sync
uv run python main.py
```

The server starts on `http://localhost:8000` and uses [haiku.rag configuration](https://ggozad.github.io/haiku.rag/configuration/).

## Architecture

The backend uses `create_agui_server()` from `haiku.rag.graph.agui.server` which provides:

- **Research graph execution**: Multi-iteration research workflow
- **AG-UI protocol**: Server-Sent Events (SSE) streaming for real-time state updates
- **Delta state updates**: Efficient incremental state synchronization using JSON Patch operations

## Endpoints

- `GET /health` - Health check with configuration info
- `POST /agent/research/stream` - Research graph streaming endpoint (AG-UI protocol)
