# Model Context Protocol (MCP)

The MCP server exposes `haiku.rag` as MCP tools for compatible MCP clients like Claude Desktop.

## Starting MCP Server

The MCP server supports Streamable HTTP and stdio transports:

```bash
# Default streamable HTTP transport on port 8001
haiku-rag mcp

# Custom port
haiku-rag mcp --port 9000

# stdio transport (for Claude Desktop)
haiku-rag mcp --stdio

# Read-only mode (excludes write tools)
haiku-rag --read-only mcp --stdio
```

**Read-only mode:** When `--read-only` is specified, write tools (`add_document_from_file`, `add_document_from_url`, `add_document_from_text`, `delete_document`) are not registered. Only search and query tools remain available.

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

After restarting Claude Desktop, you can ask Claude to search your documents, add new content, or answer questions using your knowledge base.

## Available Tools

### Document Management

- **`add_document_from_file`** - Add documents from local file paths
  - `file_path` (required): Path to the file
  - `metadata` (optional): Key-value metadata
  - `title` (optional): Human-readable title

- **`add_document_from_url`** - Add documents from URLs
  - `url` (required): URL to fetch
  - `metadata` (optional): Key-value metadata
  - `title` (optional): Human-readable title

- **`add_document_from_text`** - Add documents from raw text content
  - `content` (required): Text content
  - `uri` (optional): URI identifier
  - `metadata` (optional): Key-value metadata
  - `title` (optional): Human-readable title

- **`get_document`** - Retrieve a document by ID
  - `document_id` (required): The document ID

- **`list_documents`** - List documents with pagination and filtering
  - `limit` (optional): Maximum number to return
  - `offset` (optional): Number to skip
  - `filter` (optional): SQL WHERE clause for filtering

- **`delete_document`** - Delete a document by ID
  - `document_id` (required): The document ID

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
  - `deep` (optional): Use multi-agent deep QA for complex questions (default: false)

- **`analyze`** - Answer complex analytical questions via code execution
  - `question` (required): The question to answer
  - `filter` (optional): SQL WHERE clause to restrict document access
  - `document` (optional): Document title/ID to pre-load (can repeat)
  - Best for aggregation, computation, and multi-document analysis

## Continuous ingestion

For continuous document ingestion (filesystem watch, S3 polling, HTTP
sources, a job queue with retries), run [`haiku-ingester`](ingester.md)
as a separate process against the same LanceDB.
