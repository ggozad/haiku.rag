# Model Context Protocol (MCP)

The MCP server exposes `haiku.rag` as MCP tools for compatible MCP clients.

## Available Tools

### Document Management

- `add_document_from_file` - Add documents from local file paths
- `add_document_from_url` - Add documents from URLs
- `add_document_from_text` - Add documents from raw text content
- `get_document` - Retrieve specific documents by ID
- `list_documents` - List all documents with pagination
- `delete_document` - Delete documents by ID

### Search

- `search_documents` - Search documents using hybrid search (vector + full-text)

## Starting MCP Server

The MCP server starts automatically with the serve command and supports Streamable HTTP and stdio transports:

```bash
# Default streamable HTTP transport
haiku-rag serve

# stdio transport (for Claude Desktop)
haiku-rag serve --stdio
```
