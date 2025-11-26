# Haiku RAG

Retrieval-Augmented Generation (RAG) library built on LanceDB.

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work with LanceDB as a local vector database. It uses LanceDB for storing embeddings and performs semantic (vector) search as well as full-text search combined through native hybrid search with Reciprocal Rank Fusion. Both open-source (Ollama) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

## Features

- **Local LanceDB**: No external servers required, supports also LanceDB cloud storage, S3, Google Cloud & Azure
- **Multiple embedding providers**: Ollama, VoyageAI, OpenAI, vLLM
- **Multiple QA providers**: Any provider/model supported by Pydantic AI
- **Native hybrid search**: Vector + full-text search with native LanceDB RRF reranking
- **Reranking**: Default search result reranking with MixedBread AI, Cohere, Zero Entropy, or vLLM
- **Question answering**: Built-in QA agents on your documents
- **Research graph (multi‑agent)**: Plan → Search → Evaluate → Synthesize with agentic AI
- **File monitoring**: Auto-index files when run as server
- **CLI & Python API**: Use from command line or Python
- **MCP server**: Expose as tools for AI assistants
- **Flexible document processing**: Local (docling) or remote (docling-serve) processing

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
# Add documents
haiku-rag add "Your content here"
haiku-rag add "Your content here" --meta author=alice --meta topic=notes
haiku-rag add-src document.pdf --meta source=manual

# Search
haiku-rag search "query"

# Search with filters
haiku-rag search "query" --filter "uri LIKE '%.pdf' AND title LIKE '%paper%'"

# Ask questions
haiku-rag ask "Who is the author of haiku.rag?"

# Ask questions with citations
haiku-rag ask "Who is the author of haiku.rag?" --cite

# Deep QA (multi-agent question decomposition)
haiku-rag ask "Who is the author of haiku.rag?" --deep --cite

# Deep QA with verbose output
haiku-rag ask "Who is the author of haiku.rag?" --deep --verbose

# Multi‑agent research (iterative plan/search/evaluate)
haiku-rag research \
  "What are the main drivers and trends of global temperature anomalies since 1990?" \
  --max-iterations 2 \
  --confidence-threshold 0.8 \
  --max-concurrency 3 \
  --verbose

# Rebuild database (re-chunk and re-embed all documents)
haiku-rag rebuild

# Start server with file monitoring
haiku-rag serve --monitor
```

To customize settings, create a `haiku.rag.yaml` config file (see [Configuration](https://ggozad.github.io/haiku.rag/configuration/)).

## Python Usage

```python
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.graph.agui import stream_graph
from haiku.rag.graph.research import (
    ResearchContext,
    ResearchDeps,
    ResearchState,
    build_research_graph,
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
    # Graph settings (provider, model, max_iterations, etc.) come from config
    graph = build_research_graph(config=Config)
    question = (
        "What are the main drivers and trends of global temperature "
        "anomalies since 1990?"
    )
    context = ResearchContext(original_question=question)
    state = ResearchState.from_config(context=context, config=Config)
    deps = ResearchDeps(client=client)

    # Blocking run (final result only)
    report = await graph.run(state=state, deps=deps)
    print(report.title)

    # Streaming progress (AG-UI events)
    async for event in stream_graph(graph, state, deps):
        if event["type"] == "STEP_STARTED":
            print(f"Starting step: {event['stepName']}")
        elif event["type"] == "ACTIVITY_SNAPSHOT":
            print(f"  {event['content']}")
        elif event["type"] == "RUN_FINISHED":
            print("\nResearch complete!\n")
            result = event["result"]
            print(result["title"])
            print(result["executive_summary"])
```

## MCP Server

Use with AI assistants like Claude Desktop:

```bash
haiku-rag serve --stdio
```

Provides tools for document management and search directly in your AI assistant.

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
- [MCP Server](https://ggozad.github.io/haiku.rag/mcp/) - Model Context Protocol integration
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) - Performance Benchmarks

mcp-name: io.github.ggozad/haiku-rag
