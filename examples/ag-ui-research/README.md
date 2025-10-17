# Haiku.rag Interactive Research Assistant

Interactive research assistant powered by **Haiku.rag**, **Pydantic AI**, and **AG-UI** protocol. Ask complex questions and watch the multi-agent research process unfold in real-time with synchronized state between backend and frontend.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama running on host (or configure another QA provider)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd haiku.rag/examples/ag-ui-research
   ```

2. **Configure environment** (optional, defaults to Ollama with gpt-oss:latest)

   ```bash
   cp .env.example .env
   # Edit .env to customize provider/model or add API keys
   ```

   See [haiku.rag configuration docs](https://ggozad.github.io/haiku.rag/configuration/) for provider setup.

3. **Prepare your knowledge base**

   Create and populate a haiku.rag database:
   ```bash
   # Create a data directory
   mkdir -p data
   # Add documents (requires haiku-rag installed locally)
   haiku-rag add "Your documents here" --db data/haiku_rag.lancedb
   # Or add from files
   haiku-rag add-src document.pdf --db data/haiku_rag.lancedb
   ```

4. **Start the application**
   ```bash
   docker compose up --build
   ```

5. **Open the application**
   - Frontend: http://localhost:3000
