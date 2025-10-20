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
- A2A agent for conversational interactions

See `docker/README.md` for setup instructions.

## A2A Security Examples

**Directory:** `a2a-security/`

Three examples showing how to add authentication to haiku.rag's A2A server:

### API Key Authentication

**File:** `a2a-security/apikey_example.py`

Simple header-based authentication suitable for internal services and development.

```bash
python examples/a2a-security/apikey_example.py /path/to/database.lancedb
```

### OAuth2 GitHub Authentication

**File:** `a2a-security/oauth2_github.py`

GitHub Personal Access Token authentication for GitHub-integrated services.

```bash
python examples/a2a-security/oauth2_github.py /path/to/database.lancedb
```

### OAuth2 Enterprise Authentication

**File:** `a2a-security/oauth2_example.py`

Full OAuth2 with JWT verification for enterprise environments.

```bash
python examples/a2a-security/oauth2_example.py /path/to/database.lancedb
```

See individual files for detailed setup instructions and usage examples.
