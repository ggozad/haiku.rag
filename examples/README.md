# haiku.rag Examples

Runnable examples.

## Docker Example

**Directory:** `docker/`

Docker Compose setup with docling-serve, continuous ingestion of a watched directory through `haiku-ingester`, and a read-only MCP server.

See `docker/README.md` for setup instructions.

## Custom Agent

**Script:** `custom_agent.py`

A conversational agent built on the RAG capability.

```bash
uv run python examples/custom_agent.py /path/to/db.lancedb
```

## Custom Agent with AG-UI Streaming

**Script:** `custom_agent_agui.py`

A Starlette app that adapts a native RAG-capable agent to AG-UI. The configuration places the database (`HAIKU_RAG_CONFIG_PATH`, or `./haiku.rag.yaml`):

```bash
uv run uvicorn examples.custom_agent_agui:app --reload --port 8000
```
