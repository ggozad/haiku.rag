# Model Context Protocol (MCP)

The MCP server exposes `haiku.rag` as MCP tools for compatible MCP clients like Claude Desktop.

## Starting MCP Server

The MCP server supports Streamable HTTP and stdio transports:

```bash
# Default streamable HTTP transport on 127.0.0.1:8001
haiku-rag mcp

# Custom port
haiku-rag mcp --port 9000

# Bind to all interfaces (e.g. inside a container)
haiku-rag mcp --host 0.0.0.0 --port 8001

# stdio transport (for Claude Desktop)
haiku-rag mcp --stdio
```

`--host` defaults to `127.0.0.1` (loopback only). Bind to `0.0.0.0` only
when you want the MCP server reachable from outside the local machine —
e.g. inside a Docker container with port mapping, or on a trusted LAN.

The server opens the database read-only. Ingestion goes through the CLI
(`haiku-rag add`, `add-src`, `delete`) or [`haiku-ingester`](ingester.md).

## Claude Desktop Integration

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "haiku-rag": {
      "command": "haiku-rag",
      "args": ["mcp", "--stdio"]
    }
  }
}
```

With a custom database path:

```json
{
  "mcpServers": {
    "haiku-rag": {
      "command": "haiku-rag",
      "args": ["mcp", "--stdio", "--db", "/path/to/database.lancedb"]
    }
  }
}
```

After restarting Claude Desktop, you can ask Claude to search your documents or answer questions using your knowledge base.

## Available Tools

### Documents

- **`get_document`** - Retrieve a document by ID
  - `document_id` (required): The document ID

- **`list_documents`** - List documents with pagination and filtering
  - `limit` (optional): Maximum number to return
  - `offset` (optional): Number to skip
  - `filter` (optional): SQL WHERE clause for filtering

### Search

- **`search_documents`** - Search using hybrid search (vector + full-text)
  - `query` (required): Search query
  - `limit` (optional): Maximum results (uses config default if not specified)
  - `include_images` (optional, default `true`): Attach base64-encoded picture bytes to picture-labeled results

- **`search_documents_by_image`** - Search using an image as the query (registered only when the configured embedder supports images)
  - `image_base64` (required): Base64-encoded image (PNG/JPEG bytes)
  - `limit` (optional): Maximum results
  - `include_images` (optional, default `true`)

### Question Answering

- **`ask_question`** - Ask questions about your documents
  - `question` (required): The question to ask
  - `cite` (optional): Include source citations (default: false)
  - `images_base64` (optional): Base64-encoded images attached to the question (requires a vision-capable QA model)

- **`analyze`** - Answer complex analytical questions via code execution
  - `question` (required): The question to answer
  - `filter` (optional): SQL WHERE clause to restrict document access
  - `images_base64` (optional): Base64-encoded images attached to the question (requires a vision-capable analysis model)
  - Best for aggregation, computation, and multi-document analysis

## Continuous ingestion

For continuous document ingestion (filesystem watch, S3 polling, HTTP
sources, a job queue with retries), run [`haiku-ingester`](ingester.md)
as a separate process against the same LanceDB.
