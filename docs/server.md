# Server Mode

The server provides automatic file monitoring and MCP functionality.

## Starting the Server

```bash
haiku-rag serve
```

Transport options:
- `--http` (default) - Streamable HTTP transport
- `--stdio` - Standard input/output transport
- `--sse` - Server-sent events transport

## File Monitoring

Set `MONITOR_DIRECTORIES` environment variable to enable automatic file monitoring:

```bash
export MONITOR_DIRECTORIES="/path/to/documents"
haiku-rag serve
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
