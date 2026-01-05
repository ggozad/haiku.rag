# Interactive Research Assistant

Research assistant powered by [haiku.rag](https://ggozad.github.io/haiku.rag/), [Pydantic Graph](https://ai.pydantic.dev/graph/), and [AG-UI](https://docs.ag-ui.com/). Ask complex questions and watch the research process unfold in real-time with human-in-the-loop control.

[Watch demo video](https://vimeo.com/1128874386)

## Features

- **Human-in-the-loop research**: Review and modify questions at decision points, then continue searching or generate report
- **Multi-iteration research graph**: Automated question decomposition and parallel search
- **Live state synchronization**: Real-time delta updates of research progress via AG-UI protocol
- **Rich reporting**: Generates comprehensive research reports with findings, conclusions, and sources

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A haiku.rag database with indexed documents
- Ollama (or configure another LLM provider)

### Setup

1. **Prepare your knowledge base**

   **Option A: Create a new database**
   ```bash
   haiku-rag init --db data/haiku_rag.lancedb
   haiku-rag add-src document.pdf --db data/haiku_rag.lancedb
   ```

   **Option B: Use an existing database**

   Set the `DB_PATH` environment variable to point to your existing haiku.rag database:
   ```bash
   # In .env file
   DB_PATH=/path/to/your/existing/haiku_rag.lancedb
   ```

   Or export it before running docker compose:
   ```bash
   export DB_PATH=/path/to/your/existing/haiku_rag.lancedb
   docker compose up --build
   ```

   The database will be mounted as read-write, so the research assistant can access all documents in your existing knowledge base.

2. **Configure haiku.rag**
   ```bash
   cp haiku.rag.yaml.example haiku.rag.yaml
   # Edit haiku.rag.yaml to customize provider/model
   ```
   See [haiku.rag configuration](https://ggozad.github.io/haiku.rag/configuration/) for details.

3. **Set API keys** (if using non-Ollama providers)
   ```bash
   cp .env.example .env
   # Edit .env to set your API keys
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   DB_PATH=/path/to/your/existing/haiku_rag.lancedb # If using an existing db.
   ```

4. **Pull the base image**
   ```bash
   docker pull ghcr.io/ggozad/haiku.rag-slim:latest
   ```

5. **Start the application**
   ```bash
   docker compose up --build
   ```

6. **Access the interface**
   - Frontend: http://localhost:3000
   - Backend health: http://localhost:8000/health

## How It Works

1. **Ask a question**: Type your research question in the chat
2. **Plan phase**: The research graph decomposes your question into targeted sub-questions
3. **Decision point**: Review the proposed questions in the right panel
   - Add new questions using the input field
   - Remove questions you don't need
   - Click **Search** to execute searches for pending questions
   - Click **Generate Report** to skip to synthesis (when you have enough answers)
4. **Research iterations**: After each search cycle, you return to a decision point where you can:
   - Review collected answers
   - Add follow-up questions based on findings
   - Continue searching or generate the final report
5. **Synthesis**: Generates a comprehensive research report with:
   - Executive summary
   - Main findings with supporting evidence
   - Conclusions and recommendations
   - Source citations

## Architecture

### Agent + Graph Pattern

This example demonstrates the **agent+graph** architecture with AG-UI client-side tool calls:

1. **Conversational Agent** (`agent.py`):
   - Pydantic AI agent handles user conversations
   - Decides when to invoke the research tool based on user intent
   - Responds directly to greetings/casual chat without tools

2. **Interactive Research Graph** (haiku.rag):
   - Multi-step research workflow invoked by the agent's tool
   - At decision points, emits AG-UI `TOOL_CALL_START/ARGS/END` events for `human_decision`
   - Waits for tool result via async queue before continuing

3. **Client-Side Tool Handling** (AG-UI pattern):
   - Frontend listens for `human_decision` tool calls via AG-UI events
   - Renders decision UI inline in chat when tool call is received
   - User decision sent directly to backend `/v1/research/stream` endpoint
   - Backend extracts tool result from messages and routes to waiting graph via async queue

4. **Shared Event Stream**:
   - `AGUIEmitter` is shared between agent and graph
   - Events from both flow through a single stream to the frontend
   - `STATE_DELTA` events sync research state to frontend in real-time

### Components

- **Backend** (Python):
  - Uses published `ghcr.io/ggozad/haiku.rag:latest` Docker image as base
  - `agent.py`: Pydantic AI agent with `run_research` tool, manages `ActiveResearch` registry
  - `main.py`: Custom AG-UI streaming endpoint, extracts tool results from messages
  - Real-time event forwarding from emitter to SSE stream

- **Frontend** (Next.js/React):
  - AG-UI protocol integration for real-time streaming
  - Handles `human_decision` tool calls with inline decision UI
  - Split-pane UI: chat on left, live research state on right
  - Tool results sent directly to backend endpoint

## Configuration

Configuration is done through `haiku.rag.yaml` (see `haiku.rag.yaml.example`):

- `research.provider`: LLM provider (default: `ollama`)
- `research.model`: Model name (default: `gpt-oss:latest`)
- `research.max_iterations`: Maximum research iterations (default: `3`)
- `research.confidence_threshold`: Confidence threshold for completion (default: `0.8`)
- `research.max_concurrency`: Parallel sub-question processing (default: `1`)
- `providers.ollama.base_url`: Ollama endpoint (default: `http://host.docker.internal:11434`)

Environment variables (see `.env.example`):

- `DB_PATH`: Path to haiku.rag database (default: `haiku_rag.lancedb`)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`: API keys for cloud providers

For full configuration options, see [haiku.rag configuration docs](https://ggozad.github.io/haiku.rag/configuration/).
