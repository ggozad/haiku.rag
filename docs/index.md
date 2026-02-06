# haiku.rag

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

## Quick Start

Install haiku.rag:

```bash
uv pip install haiku.rag
```

Use from Python:

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("database.lancedb", create=True) as client:
    # Add a document
    doc = await client.create_document("Your content here")

    # Search documents
    results = await client.search("query")

    # Ask questions (returns answer and citations)
    answer, citations = await client.ask("Who is the author of haiku.rag?")
```

Or use the CLI:

```bash
haiku-rag add "Your document content"
haiku-rag add "Your document content" --meta author=alice
haiku-rag add-src /path/to/document.pdf --title "Q3 Financial Report" --meta source=manual
haiku-rag search "query"
haiku-rag ask "Who is the author of haiku.rag?"
haiku-rag chat  # Interactive conversation mode
```

## Documentation

- [Getting started](tutorial.md) - Tutorial
- [Installation](installation.md) - Install haiku.rag with different providers
- [Architecture](architecture.md) - System overview and data flow
- [Configuration](configuration/index.md) - Environment variables and settings
- [CLI](cli.md) - Command line interface usage
- [Python](python.md) - Python API reference
- [Custom Pipelines](custom-pipelines.md) - Build custom processing workflows
- [Agents](agents.md) - QA, chat, and research agents
- [RLM Agent](rlm.md) - Complex analytical tasks via code execution
- [Applications](apps.md) - Chat TUI, web app, and inspector
- [Server](server.md) - File monitoring and server mode
- [MCP](mcp.md) - Model Context Protocol integration
- [Remote processing](remote-processing.md) - Remote document processing with docling-serve

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/ggozad/haiku.rag/main/LICENSE).
