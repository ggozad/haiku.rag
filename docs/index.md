---
title: haiku.rag
description: Local-first agentic RAG. Index PDFs, web pages, and whole directories, then ask questions and get answers cited to page numbers and section headings. Hybrid search, reranking, and multimodal retrieval on embedded LanceDB.
---

haiku.rag indexes PDFs, web pages, and whole directories, retrieves with hybrid search, and answers with citations down to the page number and section heading. It runs on an embedded database with open models, so your documents stay on your machine and there is no server to operate.

```bash
uv pip install haiku.rag

haiku-rag init
haiku-rag add-src ~/Documents/some-paper.pdf
haiku-rag ask "what does it conclude?"
```

[Quickstart](tutorial.md) covers provider setup and the first ingestion.

## Why haiku.rag

**Answers you can check.** Every answer carries citations with page numbers and section headings. Visual grounding shows the cited chunk highlighted on the original page image. Optional capabilities require an answer to declare what grounds it, including declaring that nothing does.

**Local-first, no server.** Embedded [LanceDB](https://lancedb.com/) and open models through [Ollama](https://ollama.com/) by default. No database to run and no API keys required. The same code runs against S3, GCS, Azure, LanceDB Cloud, or any provider Pydantic AI supports.

**Built for agents.** Native [Pydantic AI](https://ai.pydantic.dev/) capabilities compose into your own agents. An [MCP server](mcp.md) exposes the same database to Claude Desktop and other assistants. The analysis capability runs sandboxed Python across documents for questions that need computation rather than retrieval.

**Measured, not asserted.** Retrieval and answer quality are tracked against public benchmarks with runnable configs. See [Benchmarks](benchmarks.md).

## Start here

- [Quickstart](tutorial.md): install, index, chat.
- [Installation](installation.md): packages and extras.
- [Architecture](overview.md): how a document becomes a cited answer.
- [Capabilities](capabilities/index.md): native RAG and analysis capabilities for Pydantic AI agents.
- [Python API](python.md): use haiku.rag from code.
- [MCP server](mcp.md): expose haiku.rag to Claude Desktop or other AI assistants.
- [Configuration](configuration/index.md): every setting.

MIT licensed. Source on [GitHub](https://github.com/ggozad/haiku.rag).
