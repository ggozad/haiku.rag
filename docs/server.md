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
- Quarto Markdown (`.qmd`)
- R Markdown (`.rmd`)
- LaTeX (`.tex`, `.latex`)
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
- PlantUML (`.puml`, `.plantuml`, `.pu`)
- And other text-based code files

**Plain text:**
- Text files (`.txt`)
- RST (`.rst`)

URLs are also supported - the content is fetched and converted to markdown.

## S3 / Object Storage Monitoring

The server can also poll S3-compatible object storage (AWS S3, SeaweedFS, MinIO, Cloudflare R2, etc.) for new, modified, and deleted objects, treating each one as a document source.

Install the optional `[s3]` extra:

```bash
pip install haiku.rag-slim[s3]
# or, for the full package:
pip install haiku.rag[s3]
```

Configure one or more S3 sources under `monitor.s3` in `haiku.rag.yaml`:

```yaml
monitor:
  s3:
    - uri: s3://my-bucket/incoming/
      poll_interval: 300        # seconds between sweeps; default 300
      include_patterns: ["*.pdf", "*.md"]
      ignore_patterns: ["draft*"]
      delete_orphans: true
      storage_options:
        endpoint: http://seaweed:8333
        aws_access_key_id: ${AWS_KEY}
        aws_secret_access_key: ${AWS_SECRET}
        region: us-east-1
        allow_http: "true"
```

Then start the server with `--monitor` — the same flag enables both local-directory and S3 watchers:

```bash
haiku-rag serve --monitor
```

Each entry in `monitor.s3` runs as its own polling task. On every sweep the watcher lists all objects under the configured prefix, compares each object's S3 ETag against the document's stored `metadata["etag"]`, and only re-fetches keys whose ETag has changed. When the bytes turn out to match the stored MD5 (e.g. the same file was re-uploaded with a different multipart chunk size), the watcher refreshes the etag and skips re-chunking. Otherwise the document is downloaded, chunked, and re-embedded.

### Credentials

`storage_options` follows the same convention as `lancedb.storage_options` — the dict is passed straight to obstore (the same Rust `object_store` library LanceDB uses internally), so any keys you've configured there work here too. When `storage_options` is omitted, the watcher falls back to the AWS default credential chain (environment variables, IAM instance role, AWS profile).

### Orphan deletion scope

`delete_orphans: true` is per-entry: a watcher only removes documents whose URI starts with that entry's `s3://bucket/prefix/`. Documents from other buckets, prefixes, or local-file sources are never touched.

### One-off ingestion

`s3://` URIs are also a first-class source for `haiku-rag add-src` and the MCP `add_document_from_url` tool — see [CLI → Add Documents](cli.md#add-documents).
