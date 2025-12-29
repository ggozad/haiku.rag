# Haiku RAG

[![Tests](https://github.com/ggozad/haiku.rag/actions/workflows/test.yml/badge.svg)](https://github.com/ggozad/haiku.rag/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ggozad/haiku.rag/graph/badge.svg)](https://codecov.io/gh/ggozad/haiku.rag)

Agentic RAG built on [LanceDB](https://lancedb.com/), [Pydantic AI](https://ai.pydantic.dev/), and [Docling](https://docling-project.github.io/docling/).

## Features

- **Hybrid search** — Vector + full-text with Reciprocal Rank Fusion
- **Reranking** — MxBAI, Cohere, Zero Entropy, or vLLM
- **Question answering** — QA agents with citations (page numbers, section headings)
- **Research agents** — Multi-agent workflows via pydantic-graph: plan, search, evaluate, synthesize
- **Document structure** — Stores full [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/), enabling structure-aware context expansion
- **Visual grounding** — View chunks highlighted on original page images
- **Time travel** — Query the database at any historical point with `--before`
- **Multiple providers** — Embeddings: Ollama, OpenAI, VoyageAI, LM Studio, vLLM. QA/Research: any model supported by Pydantic AI
- **Local-first** — Embedded LanceDB, no servers required. Also supports S3, GCS, Azure, and LanceDB Cloud
- **MCP server** — Expose as tools for AI assistants (Claude Desktop, etc.)
- **File monitoring** — Watch directories and auto-index on changes
- **Inspector** — TUI for browsing documents, chunks, and search results
- **CLI & Python API** — Full functionality from command line or code

## Installation

**Python 3.12 or newer required**

### Full Package (Recommended)

```bash
uv pip install haiku.rag
```

Includes all features: document processing, all embedding providers, and rerankers.

### Slim Package (Minimal Dependencies)

```bash
uv pip install haiku.rag-slim
```

Install only the extras you need. See the [Installation](https://ggozad.github.io/haiku.rag/installation/) documentation for available options

## Quick Start

```bash
# Index a PDF
haiku-rag add-src paper.pdf

# Search
haiku-rag search "attention mechanism"

# Ask questions with citations
haiku-rag ask "What datasets were used for evaluation?" --cite

# Deep QA — decomposes complex questions into sub-queries
haiku-rag ask "How does the proposed method compare to the baseline on MMLU?" --deep

# Research mode — iterative planning and search
haiku-rag research "What are the limitations of the approach?" --verbose

# Interactive research — human-in-the-loop with decision points
haiku-rag research "Compare the approaches discussed" --interactive

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

For research agents and streaming with [AG-UI](https://docs.ag-ui.com/), see the [Agents docs](https://ggozad.github.io/haiku.rag/agents/).

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

- **[Interactive Research Assistant](examples/ag-ui-research/)** - Full-stack research assistant with Pydantic AI and AG-UI featuring human-in-the-loop approval and real-time state synchronization
- **[Docker Setup](examples/docker/)** - Complete Docker deployment with file monitoring and MCP server
- **[A2A Server](examples/a2a-server/)** - Self-contained A2A protocol server package with conversational agent interface

## Documentation

Full documentation at: https://ggozad.github.io/haiku.rag/

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Provider setup
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - YAML configuration
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
- [Agents](https://ggozad.github.io/haiku.rag/agents/) - QA agent and multi-agent research
- [Server](https://ggozad.github.io/haiku.rag/server/) - File monitoring, MCP, and AG-UI
- [MCP](https://ggozad.github.io/haiku.rag/mcp/) - Model Context Protocol integration
- [Inspector](https://ggozad.github.io/haiku.rag/inspector/) - Database browser TUI
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) - Performance benchmarks
- [Changelog](https://ggozad.github.io/haiku.rag/changelog/) - Version history

mcp-name: io.github.ggozad/haiku-rag
