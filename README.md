# Haiku SQLite RAG

Retrieval-Augmented Generation (RAG) library on SQLite.

`haiku.rag` is a Retrieval-Augmented Generation (RAG) library built to work on SQLite alone without the need for external vector databases. It uses [sqlite-vec](https://github.com/asg017/sqlite-vec) for storing the embeddings and performs semantic (vector) search as well as full-text search combined through Reciprocal Rank Fusion. Both open-source (Ollama) as well as commercial (OpenAI, VoyageAI) embedding providers are supported.

## Features

- **Local SQLite**: No external servers required
- **Multiple embedding providers**: Ollama, VoyageAI, OpenAI
- **Multiple QA providers**: Ollama, OpenAI, Anthropic
- **Hybrid search**: Vector + full-text search with Reciprocal Rank Fusion
- **Question answering**: Built-in QA agents on your documents
- **File monitoring**: Auto-index files when run as server
- **40+ file formats**: PDF, DOCX, HTML, Markdown, audio, URLs
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

# Rebuild database (re-chunk and re-embed all documents)
haiku-rag rebuild

# Start server with file monitoring
export MONITOR_DIRECTORIES="/path/to/docs"
haiku-rag serve
```

## Python Usage

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("database.db") as client:
    # Add document
    doc = await client.create_document("Your content")

    # Search
    results = await client.search("query")
    for chunk, score in results:
        print(f"{score:.3f}: {chunk.content}")

    # Ask questions
    answer = await client.ask("Who is the author of haiku.rag?")
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

## Docker

``` bash
docker pull topiaruss/haiku-rag:latest
docker compose up
```

### docker-compose.yml file

 * When this runs, a documents directory will be created in this directory. You can drag files here to be indexed.

 * A database directory will be created to hold your haiku-rag.db.

 * The compose file mounts your local `~/.ollama/models` directory into which you should pull in the normal way the models you want to use. These will then be accessible inside the running container.

For production, expect to edit the docker-compose.yml file 
to set a permanent location for your models, database, and document directory. 

Consult Docker resources to understand the many possibilities.


### build-multiplatform.sh

This is a simple builder to provide a multi-platform image.
Replace "topiaruss" with your own dockerhub username, create the haiku-rag 
image folder, and issue a `docker login` command

### check (tail) the logs

```bash
docker compose logs -f
```