# Interactive Research Assistant

Research assistant powered by [haiku.rag](https://ggozad.github.io/haiku.rag/), [Pydantic Graph](https://ai.pydantic.dev/graph/), and [AG-UI](https://docs.ag-ui.com/). Ask complex questions and watch the research process unfold in real-time.

[Watch demo video](https://vimeo.com/1128874386)

## Features

- **Multi-iteration research graph**: Automated question decomposition and search
- **Intelligent evaluation**: Confidence-based decision making with automatic iteration until sufficient information is gathered
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
   mkdir -p data
   haiku-rag add "Your documents here" --db data/haiku_rag.lancedb
   # Or add from files
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

1. **Start the application**
   ```bash
   docker compose up --build
   ```

2. **Access the interface**
   - Frontend: http://localhost:3000
   - Backend health: http://localhost:8000/health

## How It Works

1. **Ask a question**: Type your research question in the chat
2. **Plan phase**: The research graph automatically:
   - Decomposes your question into targeted sub-questions
   - Gathers initial context about the topic
3. **Research iterations**: The graph autonomously:
   - Searches the knowledge base for each sub-question in parallel
   - Assesses confidence in gathered information
   - Generates new follow-up questions if needed
   - Iterates until confidence threshold is met or max iterations reached
4. **Synthesis**: Generates a comprehensive research report with:
   - Executive summary
   - Main findings with supporting evidence
   - Conclusions and recommendations
   - Source citations

## Architecture

### Agent + Graph Pattern

This example demonstrates the **agent+graph** architecture pattern:

1. **Conversational Agent** (`agent.py`):
   - Pydantic AI agent handles user conversations
   - Decides when to invoke the research tool based on user intent
   - Responds directly to greetings/casual chat without tools
   - Formats research results for the user

2. **Research Graph** (haiku.rag):
   - Multi-step research workflow invoked by the agent's tool
   - Autonomous execution with plan → search → analyze → decide → synthesize flow
   - Emits AG-UI events for real-time progress tracking

3. **Shared Event Stream**:
   - `AGUIEmitter` is shared between agent and graph
   - Events from both flow through a single stream to the frontend
   - Custom streaming endpoint (`main.py`) uses anyio memory streams for proper async handling

### Components

- **Backend** (Python):
  - Uses published `ghcr.io/ggozad/haiku.rag:latest` Docker image as base
  - `agent.py`: Pydantic AI agent with `run_research` tool
  - `main.py`: Custom AG-UI streaming endpoint with anyio memory object streams
  - Real-time event forwarding from emitter to SSE stream
  - Filters out `ACTIVITY_SNAPSHOT` events (not yet supported by CopilotKit)

- **Frontend** (Next.js/React):
  - CopilotKit for AG-UI protocol integration
  - Split-pane UI: chat on left, live research state on right
  - Real-time state synchronization via Server-Sent Events (SSE)
  - `StateDisplay` component with collapsible sections for questions and report

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
