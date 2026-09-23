# haiku.rag

[![PyPI](https://img.shields.io/pypi/v/haiku.rag)](https://pypi.org/project/haiku.rag/)
[![Python](https://img.shields.io/pypi/pyversions/haiku.rag)](https://pypi.org/project/haiku.rag/)
[![Downloads](https://static.pepy.tech/badge/haiku-rag-slim/month)](https://pepy.tech/projects/haiku-rag-slim)
[![Docs](https://img.shields.io/badge/docs-ggozad.github.io-blue)](https://ggozad.github.io/haiku.rag/)
[![Tests](https://github.com/ggozad/haiku.rag/actions/workflows/test.yml/badge.svg)](https://github.com/ggozad/haiku.rag/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ggozad/haiku.rag/graph/badge.svg)](https://codecov.io/gh/ggozad/haiku.rag)

Agentic RAG that answers questions about your own documents with citations to page numbers and section headings. It runs on an embedded LanceDB database with open models through Ollama by default, so no server or API key is needed. Any provider Pydantic AI supports works in their place, and the same database can live on S3, GCS, Azure or LanceDB Cloud.

Built on [LanceDB](https://lancedb.com/), [Pydantic AI](https://ai.pydantic.dev/) and [Docling](https://docling-project.github.io/docling/). Documentation: [ggozad.github.io/haiku.rag](https://ggozad.github.io/haiku.rag/).

## Features

- **Ingest** PDFs, office documents, HTML, Markdown and images with Docling, in-process or on docling-serve. The stored DoclingDocument keeps headings, tables, pictures and page provenance.
- **Search** with hybrid vector and full-text retrieval, optional reranking (cross-encoders, Jina, Cohere, Zero Entropy, vLLM, OpenRouter), section-aware context expansion, and image search with a multimodal embedder (vLLM, OpenRouter, VoyageAI, Cohere). Across several named databases at once.
- **Answer** with the RAG capability: it searches, runs sandboxed Python over the documents for counting and aggregation, and cites page numbers and headings. Vision models receive the figures. Optional capabilities compact earlier evidence in long conversations and require every answer to declare its grounding.
- **Check** a citation by drawing its chunk on the page image, from the CLI, the chat TUI or Python.
- **Integrate** through the Python API, native Pydantic AI capabilities, an MCP server for Claude Code, Codex and Claude Desktop, and a reference web app.
- **Operate** with the `haiku-ingester` service (filesystem, HTTP, S3 and WebDAV sources, a SQLite or Postgres job queue with retries, a control plane and dashboard), tags and rollback, vacuum, and `haiku-rag doctor` health checks.

## Installation

Python 3.12 or newer.

```bash
pip install haiku.rag        # Docling, the VoyageAI and Cohere embedders, every reranker, the TUI
pip install haiku.rag-slim   # the core, with extras chosen by you
```

The ingester, S3 access and model providers other than Ollama and OpenAI-compatible endpoints are extras. See [Installation](https://ggozad.github.io/haiku.rag/installation/).

## Quick start

The default configuration uses Ollama for embeddings and answers. The [quickstart](https://ggozad.github.io/haiku.rag/tutorial/) covers the models to pull and using OpenAI instead.

```bash
haiku-rag init                                  # create the database
haiku-rag add-src paper.pdf                     # index a file, URL or directory
haiku-rag search "attention mechanism"
haiku-rag ask "What datasets were used for evaluation?"
haiku-rag ask "How many documents mention transformers?"
haiku-rag ask "Does this figure match the spec?" --image figure.png
haiku-rag chat                                  # multi-turn chat in the terminal
```

Continuous ingestion from configured sources runs as a separate service, with the ingester extra (`pip install 'haiku.rag[ingester]'`):

```bash
haiku-ingester serve
```

## Python API

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("knowledge.lancedb", create=True) as rag:
    await rag.create_document_from_source("paper.pdf")
    await rag.create_document_from_source("https://arxiv.org/pdf/1706.03762")

    results = await rag.search("self-attention")
    for result in results:
        print(f"{result.score:.2f} | p.{result.page_numbers} | {result.content[:100]}")

    answer, citations = await rag.ask("What is the complexity of self-attention?")
    print(answer)
    for cite in citations:
        print(f"  [{cite.chunk_id}] p.{cite.page_numbers}: {cite.content[:80]}")
```

To compose your own agent, see [Capabilities](https://ggozad.github.io/haiku.rag/capabilities/).

## MCP server

```bash
haiku-rag mcp --stdio
```

The server gives an assistant search, document reading and a Python sandbox over the documents. In Claude Code, the plugin registers it with a skill:

```bash
claude plugin marketplace add ggozad/haiku.rag
claude plugin install haiku-rag
```

Codex and Claude Desktop setup is in the [MCP docs](https://ggozad.github.io/haiku.rag/mcp/).

## Examples

- [Docker setup](https://github.com/ggozad/haiku.rag/tree/main/examples/docker): docling-serve, the ingester and the MCP server
- [Web application](https://github.com/ggozad/haiku.rag/tree/main/app): conversational RAG over AG-UI with a CopilotKit frontend

## Documentation

- [Quickstart](https://ggozad.github.io/haiku.rag/tutorial/): install, index, chat
- [Installation](https://ggozad.github.io/haiku.rag/installation/): packages and extras
- [Architecture](https://ggozad.github.io/haiku.rag/overview/): how a document becomes a cited answer
- [CLI](https://ggozad.github.io/haiku.rag/cli/) and [Chat and inspector](https://ggozad.github.io/haiku.rag/chat/)
- [Capabilities](https://ggozad.github.io/haiku.rag/capabilities/): native Pydantic AI capabilities
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/): every setting, and [tuning](https://ggozad.github.io/haiku.rag/tuning/)
- [Ingester](https://ggozad.github.io/haiku.rag/ingester/), [MCP](https://ggozad.github.io/haiku.rag/mcp/) and [remote processing](https://ggozad.github.io/haiku.rag/remote-processing/)
- [Python API](https://ggozad.github.io/haiku.rag/python/) and [custom pipelines](https://ggozad.github.io/haiku.rag/custom-pipelines/)
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) and the [changelog](https://ggozad.github.io/haiku.rag/changelog/)

## License

[MIT](https://github.com/ggozad/haiku.rag/blob/main/LICENSE).

<!-- mcp-name is used by the MCP registry to identify this server -->
mcp-name: io.github.ggozad/haiku-rag
