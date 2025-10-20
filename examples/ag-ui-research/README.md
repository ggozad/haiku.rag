# Interactive Research Assistant

Research assistant powered by [haiku.rag](https://ggozad.github.io/haiku.rag/), [Pydantic AI](https://ai.pydantic.dev/), and [AG-UI](https://docs.ag-ui.com/). Ask complex questions and watch the research process unfold in real-time.

## Features

- **Multi-step research workflow**: Question decomposition, search, analysis, and synthesis
- **Human-in-the-loop**: Approve or revise research plans before execution
- **Live state synchronization**: Real-time updates of research progress between backend and frontend
- **Context expansion**: Automatically expands top search results for better context
- **Rich reporting**: Generates structured reports with findings, conclusions, and citations

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A haiku.rag database with indexed documents
- Ollama (or configure another LLM provider)

### Setup

1. **Prepare your knowledge base**
   ```bash
   mkdir -p data
   haiku-rag add "Your documents here" --db data/haiku_rag.lancedb
   # Or add from files
   haiku-rag add-src document.pdf --db data/haiku_rag.lancedb
   ```

2. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env to customize provider/model
   ```
   See [haiku.rag configuration](https://ggozad.github.io/haiku.rag/configuration/) for details.

3. **Start the application**
   ```bash
   docker compose up --build
   ```

4. **Access the interface**
   - Frontend: http://localhost:3000
   - Backend health: http://localhost:8000/health

## How It Works

1. **Ask a question**: Type your research question in the chat
2. **Review the plan**: The agent decomposes your question into 3 sub-questions
3. **Approve or revise**: Choose to approve the plan or request changes
4. **Watch it work**: The agent automatically:
   - Searches the knowledge base for each sub-question
   - Extracts key insights from search results
   - Evaluates overall confidence in findings
5. **Get your report**: Receive a structured research report with citations

## Architecture

- **Backend** (Python): Pydantic AI agent with haiku.rag integration
  - `agent.py`: Research agent with tool definitions
  - `main.py`: Starlette app serving AG-UI protocol

- **Frontend** (Next.js): CopilotKit/AG-UI interface
  - Real-time state synchronization with backend
  - Interactive approval workflow
  - Collapsible research plan and insights display

## Configuration

Environment variables (see `.env.example`):

- `DB_PATH`: Path to haiku.rag database (default: `haiku_rag.lancedb`)
- `QA_PROVIDER`: LLM provider (default: `ollama`)
- `QA_MODEL`: Model name (default: `gpt-oss:latest`)
- `OLLAMA_BASE_URL`: Ollama endpoint (default: `http://host.docker.internal:11434`)

For other providers (OpenAI, Anthropic, etc.), see [haiku.rag configuration docs](https://ggozad.github.io/haiku.rag/configuration/).
