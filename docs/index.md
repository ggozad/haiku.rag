# haiku.rag

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work on SQLite alone without the need for external vector databases. It uses [sqlite-vec](https://github.com/asg017/sqlite-vec) for storing the embeddings and performs semantic (vector) search as well as full-text search combined through Reciprocal Rank Fusion. Both open-source (Ollama, MixedBread AI) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

## Features

- **Local SQLite**: No need to run additional servers
- **Support for various embedding providers**: Ollama, VoyageAI, OpenAI or add your own
- **Hybrid Search**: Vector search using `sqlite-vec` combined with full-text search `FTS5`, using Reciprocal Rank Fusion
- **Reranking**: Optional result reranking with MixedBread AI or Cohere
- **Question Answering**: Built-in QA agents using Ollama, OpenAI, or Anthropic.
- **File monitoring**: Automatically index files when run as a server
- **Extended file format support**: Parse 40+ file formats including PDF, DOCX, HTML, Markdown, code files and more. Or add a URL!
- **MCP server**: Exposes functionality as MCP tools
- **CLI commands**: Access all functionality from your terminal
- **Python client**: Call `haiku.rag` from your own python applications

## Quick Start

Install haiku.rag:
```bash
uv pip install haiku.rag
```

Use from Python:
```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("database.db") as client:
    # Add a document
    doc = await client.create_document("Your content here")

    # Search documents
    results = await client.search("query")

    # Ask questions
    answer = await client.ask("Who is the author of haiku.rag?", rerank=False)
```

Or use the CLI:
```bash
haiku-rag add "Your document content"
haiku-rag search "query"
haiku-rag ask "Who is the author of haiku.rag?"
```

## Documentation

- [Installation](installation.md) - Install haiku.rag with different providers
- [Configuration](configuration.md) - Environment variables and settings
- [CLI](cli.md) - Command line interface usage
- [Question Answering](qa.md) - QA agents and natural language queries
- [Server](server.md) - File monitoring and server mode
- [MCP](mcp.md) - Model Context Protocol integration
- [Python](python.md) - Python API reference

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/ggozad/haiku.rag/main/LICENSE).
