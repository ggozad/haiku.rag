# haiku.rag Examples

This directory contains example scripts demonstrating various features of haiku.rag.

## Docker Example

**Directory:** `docker/`

Complete Docker setup for running haiku.rag with all services:
- File monitoring for automatic document indexing
- MCP server for AI assistant integration

See `docker/README.md` for setup instructions.

## Custom Agent

**Script:** `custom_agent.py`

Uses the RAG skill with `SkillToolset` to build a conversational agent.

```bash
uv run python examples/custom_agent.py /path/to/db.lancedb
```

## Custom Agent with AG-UI Streaming

**Script:** `custom_agent_agui.py`

A Starlette app that serves an AG-UI streaming endpoint using the RAG skill with `SkillToolset`.

```bash
DB_PATH=/path/to/db.lancedb uv run uvicorn examples.custom_agent_agui:app --reload --port 8000
```
