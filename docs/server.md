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
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["*"]
```

- **host**: Bind address (default: `0.0.0.0`)
- **port**: Server port (default: `8000`)
- **cors_origins**: Allowed CORS origins (default: `["*"]`)
- **cors_credentials**: Allow credentials in CORS requests (default: `true`)
- **cors_methods**: Allowed HTTP methods (default: `["GET", "POST", "OPTIONS"]`)
- **cors_headers**: Allowed headers (default: `["*"]`)

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
- `STATE_DELTA` - Incremental state changes (JSON Patch format)
- `STEP_STARTED` - Node execution started
- `STEP_FINISHED` - Node execution completed
- `ACTIVITY_SNAPSHOT` - Progress update with structured data
- `RUN_FINISHED` - Graph execution completed with result
- `RUN_ERROR` - Error during execution

Example event output:
```
data: {"type":"RUN_STARTED","threadId":"abc123","runId":"xyz789"}

data: {"type":"STATE_SNAPSHOT","snapshot":{"context":{"original_question":"What are the key features of haiku.rag?"},"iterations":0}}

data: {"type":"STEP_STARTED","stepName":"plan"}

data: {"type":"ACTIVITY_SNAPSHOT","messageId":"msg-1","activityType":"planning","stepName":"plan","content":{"message":"Created plan with 3 sub-questions","sub_questions":["What is X?","How does Y work?","Why is Z important?"]}}

data: {"type":"STEP_FINISHED","stepName":"plan"}

data: {"type":"STATE_DELTA","delta":[{"op":"replace","path":"/iterations","value":1}]}

data: {"type":"ACTIVITY_SNAPSHOT","messageId":"msg-2","activityType":"evaluating","stepName":"decide","content":{"message":"Confidence: 85%, Sufficient: Yes","confidence":0.85,"is_sufficient":true}}

data: {"type":"RUN_FINISHED","threadId":"abc123","runId":"xyz789","result":{"title":"Research Report","executive_summary":"..."}}
```

**Activity Event Structure:**

`ACTIVITY_SNAPSHOT` events include a `content` object with:
- `message` - Human-readable progress message (always present)
- `stepName` - The graph step emitting this activity (when available)
- Additional structured fields depending on the activity type:
  - **Planning**: `sub_questions` (list of strings)
  - **Searching**: `query` (string), `confidence` (float, on completion), `error` (string, on failure)
  - **Analyzing**: `insights` (list of insight objects), `gaps` (list of gap objects), `resolved_gaps` (list of strings)
  - **Evaluating**: `confidence` (float), `is_sufficient` (boolean) for research; `is_sufficient` (boolean), `iterations` (int) for deep QA

The `message` field is always present for simple rendering, while structured fields enable richer UI features like displaying lists, charts, and detailed status information.

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
