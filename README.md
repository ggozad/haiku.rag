# Haiku SQLite RAG

A Retrieval-Augmented Generation (RAG) library on SQLite.

## Features
- **Local SQLite**: No need to run additional servers
- **Support for various embedding providers**: You can use Ollama, VoyageAI, OpenAI or add your own
- **Hybrid Search**: Vector search using `sqlite-vec` combined with full-text search `FTS5`, using Reciprocal Rank Fusion
- **File monitoring** when run as a server automatically indexing your files
- **Extended file format Support**: Parse 40+ file formats including PDF, DOCX, HTML, Markdown, audio and more. Or add a url!
- **MCP server** Exposes functionality as MCP tools.
- **CLI commands** Access all functionality from your terminal
- **Python client** Call `haiku.rag` from your own python applications.

## Installation

```bash
uv pip install haiku.rag
```

By default Ollama (with the `mxbai-embed-large` model) is used for the embeddings.
For other providers use:

- **VoyageAI**: `uv pip install haiku.rag --extra voyageai`
- **OpenAI**: `uv pip install haiku.rag --extra openai`

## Configuration

You can set the directories to monitor using the `MONITOR_DIRECTORIES` environment variable (as comma separated values) :

```bash
# Monitor single directory
export MONITOR_DIRECTORIES="/path/to/documents,/another_path/to/documents"
```

If you want to use an alternative embeddings provider (Ollama being the default) you will need to set the provider details through environment variables:

By default:

```bash
EMBEDDINGS_PROVIDER="ollama"
EMBEDDINGS_MODEL="mxbai-embed-large" # or any other model
EMBEDDINGS_VECTOR_DIM=1024
```

For VoyageAI:
```bash
EMBEDDINGS_PROVIDER="voyageai"
EMBEDDINGS_MODEL="voyage-3.5" # or any other model
EMBEDDINGS_VECTOR_DIM=1024
VOYAGE_API_KEY="your-api-key"
```

For OpenAI:
```bash
EMBEDDINGS_PROVIDER="openai"
EMBEDDINGS_MODEL="text-embedding-3-small" # or text-embedding-3-large
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="your-api-key"
```

## Command Line Interface

`haiku.rag` includes a CLI application for managing documents and performing searches from the command line:

### Available Commands

```bash
# List all documents
haiku-rag list

# Add document from text
haiku-rag add "Your document content here"

# Add document from file or URL
haiku-rag add-src /path/to/document.pdf
haiku-rag add-src https://example.com/article.html

# Get and display a specific document
haiku-rag get 1

# Delete a document by ID
haiku-rag delete 1

# Search documents
haiku-rag search "machine learning"

# Search with custom options
haiku-rag search "python programming" --limit 10 --k 100

# Start file monitoring & MCP server (default HTTP transport)
haiku-rag serve # --stdio for stdio transport or --sse for SSE transport
```

All commands support the `--db` option to specify a custom database path. Run
```bash
haiku-rag command -h
```
to see additional parameters for a command.

## File Monitoring & MCP server

You can start the server (using Streamble HTTP, stdio or SSE transports) with:

```bash
# Start with default HTTP transport
haiku-rag serve # --stdio for stdio transport or --sse for SSE transport
```

You need to have set the `MONITOR_DIRECTORIES` environment variable for monitoring to take place.

### File monitoring

`haiku.rag` can watch directories for changes and automatically update the document store:

- **Startup**: Scan all monitored directories and add any new files
- **File Added/Modified**: Automatically parse and add/update the document in the database
- **File Deleted**: Remove the corresponding document from the database

### MCP Server

`haiku.rag` includes a Model Context Protocol (MCP) server that exposes RAG functionality as tools for AI assistants like Claude Desktop. The MCP server provides the following tools:

- `add_document_from_file` - Add documents from local file paths
- `add_document_from_url` - Add documents from URLs
- `add_document_from_text` - Add documents from raw text content
- `search_documents` - Search documents using hybrid search
- `get_document` - Retrieve specific documents by ID
- `list_documents` - List all documents with pagination
- `delete_document` - Delete documents by ID

## Using `haiku.rag` from python

### Managing documents

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Use as async context manager (recommended)
async with HaikuRAG("path/to/database.db") as client:
    # Create document from text
    doc = await client.create_document(
        content="Your document content here",
        uri="doc://example",
        metadata={"source": "manual", "topic": "example"}
    )

    # Create document from file (auto-parses content)
    doc = await client.create_document_from_source("path/to/document.pdf")

    # Create document from URL
    doc = await client.create_document_from_source("https://example.com/article.html")

    # Retrieve documents
    doc = await client.get_document_by_id(1)
    doc = await client.get_document_by_uri("file:///path/to/document.pdf")

    # List all documents with pagination
    docs = await client.list_documents(limit=10, offset=0)

    # Update document content
    doc.content = "Updated content"
    await client.update_document(doc)

    # Delete document
    await client.delete_document(doc.id)

    # Search documents using hybrid search (vector + full-text)
    results = await client.search("machine learning algorithms", limit=5)
    for chunk, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {chunk.content}")
        print(f"Document ID: {chunk.document_id}")
        print("---")
```

## Searching documents

```python
async with HaikuRAG("database.db") as client:

    results = await client.search(
        query="machine learning",
        limit=5,  # Maximum results to return, defaults to 5
        k=60      # RRF parameter for reciprocal rank fusion, defaults to 60
    )

    # Process results
    for chunk, relevance_score in results:
        print(f"Relevance: {relevance_score:.3f}")
        print(f"Content: {chunk.content}")
        print(f"From document: {chunk.document_id}")
```

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