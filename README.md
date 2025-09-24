# Haiku RAG

Retrieval-Augmented Generation (RAG) library built on LanceDB.

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work with LanceDB as a local vector database. It uses LanceDB for storing embeddings and performs semantic (vector) search as well as full-text search combined through native hybrid search with Reciprocal Rank Fusion. Both open-source (Ollama) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

> **Note**: Starting with version 0.7.0, haiku.rag uses LanceDB instead of SQLite. If you have an existing SQLite database, use `haiku-rag migrate old_database.sqlite` to migrate your data safely.

## Features

- **Local LanceDB**: No external servers required, supports also LanceDB cloud storage, S3, Google Cloud & Azure
- **Multiple embedding providers**: Ollama, VoyageAI, OpenAI, vLLM
- **Multiple QA providers**: Any provider/model supported by Pydantic AI
- **Research graph (multi‑agent)**: Plan → Search → Evaluate → Synthesize with agentic AI
- **Native hybrid search**: Vector + full-text search with native LanceDB RRF reranking
- **Reranking**: Default search result reranking with MixedBread AI, Cohere, or vLLM
- **Question answering**: Built-in QA agents on your documents
- **File monitoring**: Auto-index files when run as server
- **40+ file formats**: PDF, DOCX, HTML, Markdown, code files, URLs
- **MCP server**: Expose as tools for AI assistants
- **CLI & Python API**: Use from command line or Python

## Quick Start

```bash
# Install
uv pip install haiku.rag

# Add documents
haiku-rag add "Your content here"
haiku-rag add "Your content here" --meta author=alice --meta topic=notes
haiku-rag add-src document.pdf --meta source=manual

# Search
haiku-rag search "query"

# Ask questions
haiku-rag ask "Who is the author of haiku.rag?"

# Ask questions with citations
haiku-rag ask "Who is the author of haiku.rag?" --cite

# Multi‑agent research (iterative plan/search/evaluate)
haiku-rag research \
  "What are the main drivers and trends of global temperature anomalies since 1990?" \
  --max-iterations 2 \
  --confidence-threshold 0.8 \
  --max-concurrency 3 \
  --verbose

# Rebuild database (re-chunk and re-embed all documents)
haiku-rag rebuild

# Migrate from SQLite to LanceDB
haiku-rag migrate old_database.sqlite

# Start server with file monitoring
export MONITOR_DIRECTORIES="/path/to/docs"
haiku-rag serve
```

## Python Usage

```python
from haiku.rag.client import HaikuRAG
from haiku.rag.research import (
    PlanNode,
    ResearchContext,
    ResearchDeps,
    ResearchState,
    build_research_graph,
    stream_research_graph,
)

async with HaikuRAG("database.lancedb") as client:
    # Add document
    doc = await client.create_document("Your content")

    # Search (reranking enabled by default)
    results = await client.search("query")
    for chunk, score in results:
        print(f"{score:.3f}: {chunk.content}")

    # Ask questions
    answer = await client.ask("Who is the author of haiku.rag?")
    print(answer)

    # Ask questions with citations
    answer = await client.ask("Who is the author of haiku.rag?", cite=True)
    print(answer)

    # Multi‑agent research pipeline (Plan → Search → Evaluate → Synthesize)
    graph = build_research_graph()
    question = (
        "What are the main drivers and trends of global temperature "
        "anomalies since 1990?"
    )
    state = ResearchState(
        context=ResearchContext(original_question=question),
        max_iterations=2,
        confidence_threshold=0.8,
        max_concurrency=2,
    )
    deps = ResearchDeps(client=client)

    # Blocking run (final result only)
    result = await graph.run(
        PlanNode(provider="openai", model="gpt-4o-mini"),
        state=state,
        deps=deps,
    )
    print(result.output.title)

    # Streaming progress (log/report/error events)
    async for event in stream_research_graph(
        graph,
        PlanNode(provider="openai", model="gpt-4o-mini"),
        state,
        deps,
    ):
        if event.type == "log":
            iteration = event.state.iterations if event.state else state.iterations
            print(f"[{iteration}] {event.message}")
        elif event.type == "report":
            print("\nResearch complete!\n")
            print(event.report.title)
            print(event.report.executive_summary)
```

## MCP Server

Use with AI assistants like Claude Desktop:

```bash
haiku-rag serve --stdio
```

Provides tools for document management and search directly in your AI assistant.

## Documentation

Full documentation at: https://ggozad.github.io/haiku.rag/

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Provider setup
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - Environment variables
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
- [Agents](https://ggozad.github.io/haiku.rag/agents/) - QA agent and multi-agent research
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) - Performance Benchmarks
