# Server Mode

The server provides automatic file monitoring and MCP functionality.

## Starting the Server

The `serve` command requires at least one service flag. You can enable file monitoring, MCP server, or both:

### MCP Server Only

```bash
haiku-rag serve --mcp
```

Transport options:
- Default - Streamable HTTP transport on port 8001
- `--stdio` - Standard input/output transport
- `--mcp-port` - Custom port (default: 8001)

### File Monitoring Only

```bash
haiku-rag serve --monitor
```

### Both Services

```bash
haiku-rag serve --monitor --mcp
```

This will start file monitoring and MCP server on port 8001.

## File Monitoring

Configure directories to monitor in your `haiku.rag.yaml` (see [Document Processing](configuration/processing.md#file-monitoring) for all options):

```yaml
monitor:
  directories:
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

### Filtering Files

You can filter which files to monitor using gitignore-style patterns:

```yaml
monitor:
  directories:
    - /path/to/documents

  # Ignore patterns (exclude files)
  ignore_patterns:
    - "*draft*"         # Ignore draft files
    - "temp/"           # Ignore temp directory
    - "**/archive/**"   # Ignore archive directories

  # Include patterns (whitelist files)
  include_patterns:
    - "*.md"           # Only markdown files
    - "**/docs/**"     # Files in docs directories
```

**Pattern behavior:**
- Extension filtering is applied first (only supported file types)
- Include patterns create a whitelist (if specified)
- Ignore patterns exclude files
- Both can be combined for fine-grained control

### Supported Formats

The file monitor processes documents using [Docling](https://github.com/DS4SD/docling), which supports:

**Documents:**
- PDF (`.pdf`) - with OCR support for scanned documents
- Microsoft Word (`.docx`)
- Microsoft Excel (`.xlsx`)
- Microsoft PowerPoint (`.pptx`)
- HTML (`.html`, `.htm`)
- Markdown (`.md`)
- AsciiDoc (`.adoc`, `.asciidoc`)

**Data formats:**
- CSV (`.csv`)
- JSON (`.json`)
- XML (`.xml`)

**Images (via OCR):**
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- TIFF (`.tiff`, `.tif`)
- BMP (`.bmp`)

**Code files:**
- Python (`.py`)
- JavaScript (`.js`)
- TypeScript (`.ts`)
- And other text-based code files

**Plain text:**
- Text files (`.txt`)
- RST (`.rst`)

URLs are also supported - the content is fetched and converted to markdown.
