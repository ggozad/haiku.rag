# Haiku RAG

[![Tests](https://github.com/ggozad/haiku.rag/actions/workflows/test.yml/badge.svg)](https://github.com/ggozad/haiku.rag/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ggozad/haiku.rag/graph/badge.svg)](https://codecov.io/gh/ggozad/haiku.rag)

Agentic RAG built on [LanceDB](https://lancedb.com/), [Pydantic AI](https://ai.pydantic.dev/), and [Docling](https://docling-project.github.io/docling/).

## Features

- **Hybrid search** — Vector + full-text with Reciprocal Rank Fusion
- **Question answering** — QA agents with citations (page numbers, section headings)
- **Reranking** — MxBAI, Cohere, Zero Entropy, or vLLM
- **Research agents** — Multi-agent workflows via pydantic-graph: plan, search, evaluate, synthesize
- **RLM agent** — Complex analytical tasks via sandboxed Python code execution (aggregation, computation, multi-document analysis)
- **Conversational RAG** — Chat TUI and web application for multi-turn conversations with session memory
- **Document structure** — Stores full [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/), enabling structure-aware context expansion
- **Multiple providers** — Embeddings: Ollama, OpenAI, VoyageAI, LM Studio, vLLM. QA/Research: any model supported by Pydantic AI
- **Local-first** — Embedded LanceDB, no servers required. Also supports S3, GCS, Azure, and LanceDB Cloud
- **CLI & Python API** — Full functionality from command line or code
- **MCP server** — Expose as tools for AI assistants (Claude Desktop, etc.)
- **Visual grounding** — View chunks highlighted on original page images
- **File monitoring** — Watch directories and auto-index on changes
- **Time travel** — Query the database at any historical point with `--before`
- **Inspector** — TUI for browsing documents, chunks, and search results

## Installation

**Python 3.12 or newer required**

### Full Package (Recommended)

```bash
pip install haiku.rag
```

Includes all features: document processing, all embedding providers, and rerankers.

Using [uv](https://docs.astral.sh/uv/)? `uv pip install haiku.rag`

### Slim Package (Minimal Dependencies)

```bash
pip install haiku.rag-slim
```

Install only the extras you need. See the [Installation](https://ggozad.github.io/haiku.rag/installation/) documentation for available options.

## Quick Start

> **Note**: Requires an embedding provider (Ollama, OpenAI, etc.). See the [Tutorial](https://ggozad.github.io/haiku.rag/tutorial/) for setup instructions.

```bash
# Index a PDF
haiku-rag add-src paper.pdf

# Search
haiku-rag search "attention mechanism"

# Ask questions with citations
haiku-rag ask "What datasets were used for evaluation?" --cite

# Research mode — iterative planning and search
haiku-rag research "What are the limitations of the approach?"

# RLM mode — complex analytical tasks via code execution
haiku-rag rlm "How many documents mention transformers?"

# Interactive chat — multi-turn conversations with memory
haiku-rag chat

# Watch a directory for changes
haiku-rag serve --monitor
```

See [Configuration](https://ggozad.github.io/haiku.rag/configuration/) for customization options.

## Python API

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("research.lancedb", create=True) as rag:
    # Index documents
    await rag.create_document_from_source("paper.pdf")
    await rag.create_document_from_source("https://arxiv.org/pdf/1706.03762")

    # Search — returns chunks with provenance
    results = await rag.search("self-attention")
    for result in results:
        print(f"{result.score:.2f} | p.{result.page_numbers} | {result.content[:100]}")

    # QA with citations
    answer, citations = await rag.ask("What is the complexity of self-attention?")
    print(answer)
    for cite in citations:
        print(f"  [{cite.chunk_id}] p.{cite.page_numbers}: {cite.content[:80]}")
```

For research agents and chat, see the [Agents docs](https://ggozad.github.io/haiku.rag/agents/).

## MCP Server

Use with AI assistants like Claude Desktop:

```bash
haiku-rag serve --mcp --stdio
```

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "haiku-rag": {
      "command": "haiku-rag",
      "args": ["serve", "--mcp", "--stdio"]
    }
  }
}
```

Provides tools for document management, search, QA, and research directly in your AI assistant.

## Examples

See the [examples directory](examples/) for working examples:

- **[Docker Setup](examples/docker/)** - Complete Docker deployment with file monitoring and MCP server
- **[Web Application](app/)** - Full-stack conversational RAG with CopilotKit frontend

## Documentation

Full documentation at: https://ggozad.github.io/haiku.rag/

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Provider setup
- [Architecture](https://ggozad.github.io/haiku.rag/architecture/) - System overview
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - YAML configuration
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
- [Agents](https://ggozad.github.io/haiku.rag/agents/) - QA, chat, and research agents
- [RLM Agent](https://ggozad.github.io/haiku.rag/rlm/) - Complex analytical tasks via code execution
- [Applications](https://ggozad.github.io/haiku.rag/apps/) - Chat TUI, web app, and inspector
- [Server](https://ggozad.github.io/haiku.rag/server/) - File monitoring and MCP
- [MCP](https://ggozad.github.io/haiku.rag/mcp/) - Model Context Protocol integration
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) - Performance benchmarks
- [Changelog](https://ggozad.github.io/haiku.rag/changelog/) - Version history

## License

This project is licensed under the [MIT License](LICENSE).

<!-- mcp-name is used by the MCP registry to identify this server -->
mcp-name: io.github.ggozad/haiku-rag
