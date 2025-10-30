# Command Line Interface

The `haiku-rag` CLI provides complete document management functionality.

!!! note
    All commands support:

    - `--db` - Specify custom database path
    - `-h` - Show help for specific command

    Example:
    ```bash
    haiku-rag list --db /path/to/custom.db
    haiku-rag add -h
    ```

## Document Management

### List Documents

```bash
haiku-rag list
```

### Add Documents

From text:
```bash
haiku-rag add "Your document content here"

# Attach metadata (repeat --meta for multiple entries)
haiku-rag add "Your document content here" --meta author=alice --meta topic=notes
```

From file or URL:
```bash
haiku-rag add-src /path/to/document.pdf
haiku-rag add-src https://example.com/article.html

# Optionally set a human‑readable title stored in the DB schema
haiku-rag add-src /mnt/data/doc1.pdf --title "Q3 Financial Report"

# Optionally attach metadata (repeat --meta). Values use JSON parsing if possible:
# numbers, booleans, null, arrays/objects; otherwise kept as strings.
haiku-rag add-src /mnt/data/doc1.pdf --meta source=manual --meta page_count=12 --meta published=true
```

From directory (recursively adds all supported files):
```bash
haiku-rag add-src /path/to/documents/
```

!!! note
    When adding a directory, the same content filters configured for [file monitoring](configuration.md#filtering-monitored-files) are applied. This means `ignore_patterns` and `include_patterns` from your configuration will be used to filter which files are added.

!!! note
    As you add documents to `haiku.rag` the database keeps growing. By default, LanceDB supports versioning
    of your data. Create/update operations are atomic‑feeling: if anything fails during chunking or embedding,
    the database rolls back to the pre‑operation snapshot using LanceDB table versioning. You can optimize and
    compact the database by running the [vacuum](#vacuum-optimize-and-cleanup) command.

### Get Document

```bash
haiku-rag get 3f4a...   # document ID
```

### Delete Document

```bash
haiku-rag delete 3f4a...   # document ID
haiku-rag rm 3f4a...       # alias
```

Use this when you want to change things like the embedding model or chunk size for example.

## Search

Basic search:
```bash
haiku-rag search "machine learning"
```

With options:
```bash
haiku-rag search "python programming" --limit 10
```

## Question Answering

Ask questions about your documents:
```bash
haiku-rag ask "Who is the author of haiku.rag?"
```

Ask questions with citations showing source documents:
```bash
haiku-rag ask "Who is the author of haiku.rag?" --cite
```

Use deep QA for complex questions (multi-agent decomposition):
```bash
haiku-rag ask "What are the main features and architecture of haiku.rag?" --deep --cite
```

Show verbose output with deep QA:
```bash
haiku-rag ask "What are the main features and architecture of haiku.rag?" --deep --verbose
```

The QA agent will search your documents for relevant information and provide a comprehensive answer. With `--cite`, responses include citations showing which documents were used. With `--deep`, the question is decomposed into sub-questions that are answered in parallel before synthesizing a final answer. With `--verbose` (only with `--deep`), you'll see the planning, searching, evaluation, and synthesis steps as they happen.
When available, citations use the document title; otherwise they fall back to the URI.

## Research

Run the multi-step research graph:

```bash
haiku-rag research "How does haiku.rag organize and query documents?" \
  --max-iterations 2 \
  --confidence-threshold 0.8 \
  --max-concurrency 3 \
  --verbose
```

Flags:
- `--max-iterations, -n`: maximum search/evaluate cycles (default: 3)
- `--confidence-threshold`: stop once evaluation confidence meets/exceeds this (default: 0.8)
- `--max-concurrency`: number of sub-questions searched in parallel each iteration (default: 3)
- `--verbose`: show planning, searching previews, evaluation summary, and stop reason

When `--verbose` is set the CLI also consumes the internal research stream, printing every `log` event as agents progress through planning, search, evaluation, and synthesis. If you build your own integration, call `stream_research_graph` to access the same `log`, `report`, and `error` events and render them however you like while the graph is running.

## Server

Start services (requires at least one flag):
```bash
# MCP server only (HTTP transport)
haiku-rag serve --mcp

# MCP server (stdio transport)
haiku-rag serve --mcp --stdio

# A2A server only
haiku-rag serve --a2a

# File monitoring only
haiku-rag serve --monitor

# All services
haiku-rag serve --monitor --mcp --a2a

# Custom ports
haiku-rag serve --mcp --mcp-port 9000 --a2a --a2a-port 9001
```

See [Server Mode](server.md) for details on available services.

### A2A Interactive Client

Connect to and chat with haiku.rag's A2A server:

```bash
# Connect to local server
haiku-rag a2aclient

# Connect to remote server
haiku-rag a2aclient --url https://example.com:8000
```

The interactive client provides:
- Rich markdown rendering of agent responses
- Multi-turn conversation with context
- Agent card discovery and display
- Compact artifact summaries

See [A2A documentation](a2a.md) for more details.

## Settings

View current configuration settings:
```bash
haiku-rag settings
```

## Maintenance

### Info (Read-only)

Display database metadata without upgrading or modifying it:

```bash
haiku-rag info [--db /path/to/your.lancedb]
```

Shows:
- path to the database
- stored haiku.rag version (from settings)
- embeddings provider/model and vector dimension
- number of documents
- table versions per table (documents, chunks)

At the end, a separate “Versions” section lists runtime package versions:
- haiku.rag
- lancedb
- docling

### Vacuum (Optimize and Cleanup)

Reduce disk usage by optimizing and pruning old table versions across all tables:

```bash
haiku-rag vacuum
```

**Automatic Cleanup:** Vacuum runs automatically in the background after document operations. By default, it removes versions older than 60 seconds (configurable via `storage.vacuum_retention_seconds`), preserving recent versions for concurrent connections. Manual vacuum can be useful for cleanup after bulk operations or to free disk space immediately.

### Rebuild Database

Rebuild the database by deleting all chunks & embeddings and re-indexing all documents. This is useful
when want to switch embeddings provider or model:

```bash
haiku-rag rebuild
```

### Download Models

Download required runtime models:

```bash
haiku-rag download-models
```

This command:
- Downloads Docling OCR/conversion models (no-op if already present).
- Pulls Ollama models referenced in your configuration (embeddings, QA, research, rerank).
