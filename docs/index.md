# haiku.rag

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work with LanceDB as a local vector database. It uses LanceDB for storing embeddings and performs semantic (vector) search as well as full-text search combined through native hybrid search with Reciprocal Rank Fusion. Both open-source (Ollama, MixedBread AI) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

## Features

- **Local LanceDB**: No external servers required, supports also LanceDB cloud storage, S3, Google Cloud & Azure
- **Multiple embedding providers**: Ollama, LM Studio, VoyageAI, OpenAI, vLLM
- **Multiple QA providers**: Any provider/model supported by Pydantic AI (Ollama, LM Studio, OpenAI, Anthropic, etc.)
- **Native hybrid search**: Vector + full-text search with native LanceDB RRF reranking
- **Reranking**: Optional result reranking with MixedBread AI, Cohere, Zero Entropy, or vLLM
- **Question answering**: Built-in QA agents on your documents
- **Research graph (multi‑agent)**: Plan → Search → Evaluate → Synthesize with agentic AI
- **File monitoring**: Auto-index files when run as server
- **Extended file format support**: Parse PDF, DOCX, HTML, Markdown, images, code files and more
- **Flexible document processing**: Local processing with docling or remote with [docling-serve](remote-processing.md)
- **MCP server**: Expose as tools for AI assistants
- **CLI & Python API**: Use from command line or Python

## Quick Start

Install haiku.rag:

```bash
uv pip install haiku.rag
```

Use from Python:

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("database.lancedb") as client:
    # Add a document
    doc = await client.create_document("Your content here")

    # Search documents
    results = await client.search("query")

    # Ask questions
    answer = await client.ask("Who is the author of haiku.rag?")
```

Or use the CLI:

```bash
haiku-rag add "Your document content"
haiku-rag add "Your document content" --meta author=alice
haiku-rag add-src /path/to/document.pdf --title "Q3 Financial Report" --meta source=manual
haiku-rag search "query"
haiku-rag ask "Who is the author of haiku.rag?"
```

## Documentation

- [Getting started](tutorial.md) - Tutorial
- [Installation](installation.md) - Install haiku.rag with different providers
- [Configuration](configuration/index.md) - Environment variables and settings
- [CLI](cli.md) - Command line interface usage
- [Server](server.md) - File monitoring and server mode
- [MCP](mcp.md) - Model Context Protocol integration
- [Python](python.md) - Python API reference
- [Agents](agents.md) - QA agent and multi-agent research
- [Remote processing](remote-processing.md) - Remote document processing with docling-serve

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/ggozad/haiku.rag/main/LICENSE).
