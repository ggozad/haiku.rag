# Server Mode

The server provides automatic file monitoring, MCP functionality, and A2A agent support.

## Starting the Server

The `serve` command requires at least one service flag. You can enable file monitoring, MCP server, A2A server, or any combination:

### MCP Server Only

```bash
haiku-rag serve --mcp
```

Transport options:
- Default - Streamable HTTP transport on port 8001
- `--stdio` - Standard input/output transport
- `--mcp-port` - Custom port (default: 8001)

### A2A Server Only

```bash
haiku-rag serve --a2a
```

Options:
- `--a2a-host` - Host to bind to (default: 127.0.0.1)
- `--a2a-port` - Port to bind to (default: 8000)

See [A2A documentation](a2a.md) for details on the conversational agent.

### File Monitoring Only

```bash
haiku-rag serve --monitor
```

### All Services

```bash
haiku-rag serve --monitor --mcp --a2a
```

This will start file monitoring, MCP server on port 8001, and A2A server on port 8000.

## File Monitoring

Configure directories to monitor in your `haiku.rag.yaml`:

```yaml
storage:
  monitor_directories:
    - /path/to/documents
    - /another/path
```

Then start the server:

```bash
haiku-rag serve --monitor
```

### Monitoring Features

- **Startup**: Scans all monitored directories and adds new files
- **File Added/Modified**: Automatically parses and updates documents
- **File Deleted**: Removes corresponding documents from database

### Supported Formats

The server can parse 40+ file formats including:
- PDF documents
- Microsoft Office (DOCX, XLSX, PPTX)
- HTML and Markdown
- Plain text files
- Code files (Python, JavaScript, etc.)
- Images (processed via OCR)
- And more...

URLs are also supported for web content.
