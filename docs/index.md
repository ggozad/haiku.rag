# haiku.rag

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work with LanceDB as a local vector database. It uses LanceDB for storing embeddings and performs semantic (vector) search as well as full-text search combined through native hybrid search with Reciprocal Rank Fusion. Both open-source (Ollama, MixedBread AI) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

> **Note**: Starting with version 0.7.0, haiku.rag uses LanceDB instead of SQLite. If you have an existing SQLite database, use `haiku-rag migrate old_database.sqlite` to migrate your data safely.

## Features

- **Local LanceDB**: No need to run additional servers
- **Support for various embedding providers**: Ollama, VoyageAI, OpenAI or add your own
- **Native Hybrid Search**: Vector search combined with full-text search using native LanceDB RRF reranking
- **Reranking**: Optional result reranking with MixedBread AI or Cohere
- **Question Answering**: Built-in QA agents using Ollama, OpenAI, or Anthropic.
- **File monitoring**: Automatically index files when run as a server
- **Extended file format support**: Parse 40+ file formats including PDF, DOCX, HTML, Markdown, code files and more. Or add a URL!
- **MCP server**: Exposes functionality as MCP tools
- **CLI commands**: Access all functionality from your terminal
  - Add sources from text, files, or URLs, optionally with a human‑readable title
- **Python client**: Call `haiku.rag` from your own python applications

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
haiku-rag migrate old_database.sqlite  # Migrate from SQLite
```

## Documentation

- [Installation](installation.md) - Install haiku.rag with different providers
- [Configuration](configuration.md) - Environment variables and settings
- [CLI](cli.md) - Command line interface usage
- [Server](server.md) - File monitoring and server mode
- [MCP](mcp.md) - Model Context Protocol integration
- [Python](python.md) - Python API reference
- [Agents](agents.md) - QA agent and multi-agent research

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/ggozad/haiku.rag/main/LICENSE).
