# Haiku RAG

Retrieval-Augmented Generation (RAG) library built on LanceDB.

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work with LanceDB as a local vector database. It uses LanceDB for storing embeddings and performs semantic (vector) search as well as full-text search combined through native hybrid search with Reciprocal Rank Fusion. Both open-source (Ollama) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

> **Note**: Starting with version 0.7.0, haiku.rag uses LanceDB instead of SQLite. If you have an existing SQLite database, use `haiku-rag migrate old_database.sqlite` to migrate your data safely.

## Features

- **Local LanceDB**: No external servers required, supports also LanceDB cloud storage, S3, Google Cloud & Azure
- **Multiple embedding providers**: Ollama, VoyageAI, OpenAI, vLLM
- **Multiple QA providers**: Any provider/model supported by Pydantic AI
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
haiku-rag add-src document.pdf

# Search
haiku-rag search "query"

# Ask questions
haiku-rag ask "Who is the author of haiku.rag?"

# Ask questions with citations
haiku-rag ask "Who is the author of haiku.rag?" --cite

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
- [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) - Performance Benchmarks
