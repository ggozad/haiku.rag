# Server Mode

The server provides automatic file monitoring, MCP functionality, and AG-UI graph streaming.

## Starting the Server

The `serve` command requires at least one service flag. You can enable file monitoring, MCP server, AG-UI server, or any combination:

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

Configure directories to monitor in your `haiku.rag.yaml`:

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

The server can parse 40+ file formats including:
- PDF documents
- Microsoft Office (DOCX, XLSX, PPTX)
- HTML and Markdown
- Plain text files
- Code files (Python, JavaScript, etc.)
- Images (processed via OCR)
- And more...

URLs are also supported for web content.

## AG-UI Server

The AG-UI server provides HTTP streaming of both research and deep ask graph execution using Server-Sent Events (SSE).

### Starting the AG-UI Server

```bash
haiku-rag serve --agui
```

This starts an HTTP server (default: http://0.0.0.0:8000) that exposes:

- `GET /health` - Health check endpoint
- `POST /v1/research/stream` - Research graph streaming endpoint
- `POST /v1/deep-ask/stream` - Deep ask graph streaming endpoint

### Configuration

Configure the AG-UI server in your `haiku.rag.yaml`:

```yaml
agui:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  cors_credentials: true
```

See [Configuration](configuration.md#ag-ui-server-configuration) for all available options.

### Using the Streaming Endpoints

Both endpoints accept POST requests with the same AG-UI RunAgentInput format and stream AG-UI events.

**Request format:**
```json
{
  "threadId": "optional-thread-id",
  "runId": "optional-run-id",
  "state": {
    "question": "What are the key features of haiku.rag?"
  },
  "messages": [],
  "config": {}
}
```

**Research endpoint example:**
```bash
curl -X POST http://localhost:8000/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "question": "What are the key features of haiku.rag?"
    }
  }' \
  --no-buffer
```

**Deep ask endpoint example:**
```bash
curl -X POST http://localhost:8000/v1/deep-ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "question": "How does haiku.rag handle document chunking?",
      "use_citations": true
    }
  }' \
  --no-buffer
```

The `--no-buffer` flag ensures curl displays events as they arrive instead of buffering them.

**Note:** The `state` object can include:
- `question`: The question to answer (required)
- `use_citations`: Enable citations in deep ask responses (optional, deep ask only)

**Response:** Server-Sent Events stream with AG-UI protocol events:
- `RUN_STARTED` - Graph execution started
- `STATE_SNAPSHOT` - Current state snapshot
- `STEP_STARTED` - Node execution started
- `STEP_FINISHED` - Node execution completed
- `ACTIVITY_SNAPSHOT` - Progress update
- `RUN_FINISHED` - Graph execution completed with result
- `RUN_ERROR` - Error during execution

Example event output:
```
data: {"type":"RUN_STARTED","threadId":"abc123","runId":"xyz789"}

data: {"type":"STATE_SNAPSHOT","snapshot":{"context":{"original_question":"What are the key features of haiku.rag?"},"iterations":0}}

data: {"type":"STEP_STARTED","stepName":"plan"}

data: {"type":"ACTIVITY_SNAPSHOT","messageId":"msg-1","activityType":"planning","content":"Creating research plan"}

data: {"type":"STEP_FINISHED","stepName":"plan"}

data: {"type":"RUN_FINISHED","threadId":"abc123","runId":"xyz789","result":{"title":"Research Report","executive_summary":"..."}}
```

The endpoint follows the [AG-UI protocol](https://docs.ag-ui.com/concepts/events) for event streaming.

### Running Multiple Services

You can run any combination of services:

```bash
# File monitoring + AG-UI
haiku-rag serve --monitor --agui

# MCP + AG-UI
haiku-rag serve --mcp --agui

# All services
haiku-rag serve --monitor --mcp --agui
```
