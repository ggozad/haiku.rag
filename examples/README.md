# haiku.rag Examples

This directory contains example scripts demonstrating various features of haiku.rag.

## Interactive Research Assistant

**Directory:** `ag-ui-research/`

Full-stack research assistant with interactive UI powered by Pydantic AI and AG-UI:
- Multi-step research workflow with question decomposition
- Human-in-the-loop approval for research plans
- Real-time state synchronization between backend and frontend
- Context expansion and insight extraction
- Structured research reports with citations

See `ag-ui-research/README.md` for setup instructions.

## Docker Example

**Directory:** `docker/`

Complete Docker setup for running haiku.rag with all services:
- File monitoring for automatic document indexing
- MCP server for AI assistant integration

See `docker/README.md` for setup instructions.

## A2A Server

**Directory:** `a2a-server/`

Self-contained A2A (Agent-to-Agent) protocol server package that provides a conversational agent interface with its own CLI and dependencies.

Features:
- Conversational context with multi-turn dialogue support
- Interactive CLI client for testing
- Security examples (API key, OAuth2 with GitHub, enterprise OAuth2)
- Full documentation and installation instructions

See `a2a-server/README.md` for complete setup and usage instructions.

Install locally:
```bash
cd a2a-server
uv sync
```
