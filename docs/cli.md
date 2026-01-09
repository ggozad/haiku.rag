# Command Line Interface

The `haiku-rag` CLI provides complete document management functionality.

!!! note
    Global options (must be specified before the command):

    - `--config` - Specify custom configuration file
    - `--read-only` - Open database in read-only mode (blocks writes, skips upgrades)
    - `--before` - Query database as it existed before a datetime (implies `--read-only`)
    - `--version` / `-v` - Show version and exit

    Per-command options:

    - `--db` - Specify custom database path
    - `-h` - Show help for specific command

    Example:
    ```bash
    haiku-rag --config /path/to/config.yaml list
    haiku-rag --config /path/to/config.yaml list --db /path/to/custom.db
    haiku-rag --read-only search "query"
    haiku-rag --before "2025-01-15" search "query"
    haiku-rag add -h
    ```

## Document Management

### List Documents

```bash
haiku-rag list
```

Filter documents by properties:
```bash
# Filter by URI pattern
haiku-rag list --filter "uri LIKE '%arxiv%'"

# Filter by exact title
haiku-rag list --filter "title = 'My Document'"

# Combine multiple conditions
haiku-rag list --filter "uri LIKE '%.pdf' AND title LIKE '%paper%'"
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
    When adding a directory, the same content filters configured for [file monitoring](configuration/processing.md#filtering-monitored-files) are applied. This means `ignore_patterns` and `include_patterns` from your configuration will be used to filter which files are added.

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

## Visualize Chunk

Display visual grounding for a chunk - shows page images with highlighted bounding boxes:

```bash
haiku-rag visualize <chunk_id>
```

This renders the source document pages with the chunk's location highlighted. Useful for verifying chunk boundaries and understanding document structure.

!!! note
    Requires a terminal with image support (iTerm2, Kitty, WezTerm, etc.) and documents processed with docling that have page images stored.

## Search

Basic search:
```bash
haiku-rag search "machine learning"
```

With options:
```bash
haiku-rag search "python programming" --limit 10
```

With filters (filter by document properties):
```bash
# Filter by URI pattern
haiku-rag search "neural networks" --filter "uri LIKE '%arxiv%'"

# Filter by exact title
haiku-rag search "transformers" --filter "title = 'Deep Learning Guide'"

# Combine multiple conditions
haiku-rag search "AI" --filter "uri LIKE '%.pdf' AND title LIKE '%paper%'"
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

Filter to specific documents:
```bash
haiku-rag ask "What are the main findings?" --filter "uri LIKE '%paper%'"
```

The QA agent searches your documents for relevant information and provides a comprehensive answer. When available, citations use the document title; otherwise they fall back to the URI.

Flags:

- `--cite`: Include citations showing which documents were used
- `--deep`: Decompose the question into sub-questions answered in parallel before synthesizing a final answer
- `--filter`: Restrict searches to documents matching the filter (see [Filtering Search Results](python.md#filtering-search-results))

## Research

Run the multi-step research graph:

```bash
haiku-rag research "How does haiku.rag organize and query documents?"
```

Filter to specific documents:

```bash
haiku-rag research "What are the key findings?" --filter "uri LIKE '%paper%'"
```

Flags:

- `--filter`: SQL WHERE clause to filter documents (see [Filtering Search Results](python.md#filtering-search-results))

Research parameters like `max_iterations`, `confidence_threshold`, and `max_concurrency` are configured in your [configuration file](configuration/index.md) under the `research` section.

## Server

Start services (requires at least one flag):
```bash
# MCP server only (HTTP transport)
haiku-rag serve --mcp

# MCP server (stdio transport)
haiku-rag serve --mcp --stdio

# File monitoring only
haiku-rag serve --monitor

# Both services
haiku-rag serve --monitor --mcp

# Custom MCP port
haiku-rag serve --mcp --mcp-port 9000

# Read-only mode (excludes write MCP tools, disables monitor)
haiku-rag --read-only serve --mcp
```

See [Server Mode](server.md) for details on available services.

## Settings

View current configuration settings:
```bash
haiku-rag settings
```

## Database Management

### Initialize Database

Create a new database:

```bash
haiku-rag init [--db /path/to/your.lancedb]
```

This creates the database with the configured settings. **All other commands require an existing database** - they will fail with an informative error if the database doesn't exist.

### Info

Display database metadata:

```bash
haiku-rag info [--db /path/to/your.lancedb]
```

Shows:
- path to the database
- stored haiku.rag version (from settings)
- embeddings provider/model and vector dimension
- number of documents and chunks (with storage sizes)
- vector index status (exists/not created, indexed/unindexed chunks)
- table versions per table (documents, chunks)

At the end, a separate "Versions" section lists runtime package versions:
- haiku.rag
- lancedb
- docling

### Create Vector Index

Create a vector index on the chunks table for fast approximate nearest neighbor search:

```bash
haiku-rag create-index [--db /path/to/your.lancedb]
```

**Requirements:**
- Minimum 256 chunks required for index creation (LanceDB training data requirement)
- Creates an IVF_PQ index using the configured `search.vector_index_metric` (cosine/l2/dot)

**When to use:**
- After ingesting documents (indexes are not created automatically)
- After adding significant new data to rebuild the index
- Use `haiku-rag info` to check index status and see how many chunks are indexed/unindexed

**Search behavior:**
- Without index: Brute-force kNN search (exact nearest neighbors, slower for large datasets)
- With index: Fast ANN (approximate nearest neighbors) using IVF_PQ
- With stale index: LanceDB combines indexed results (fast ANN) + brute-force kNN on unindexed rows
- Performance degrades as more unindexed data accumulates

### Vacuum (Optimize and Cleanup)

Reduce disk usage by optimizing and pruning old table versions across all tables:

```bash
haiku-rag vacuum
```

**Automatic Cleanup:** Vacuum runs automatically in the background after document operations. By default, it removes versions older than 1 day (configurable via `storage.vacuum_retention_seconds`), preserving recent versions for concurrent connections. Manual vacuum can be useful for cleanup after bulk operations or to free disk space immediately.

### Rebuild Database

Rebuild the database by re-indexing documents. Useful when switching embeddings provider/model or changing chunking settings:

```bash
# Full rebuild (default) - re-converts from source files, re-chunks, re-embeds
haiku-rag rebuild

# Re-chunk from stored content (no source file access)
haiku-rag rebuild --rechunk

# Only regenerate embeddings (fastest, keeps existing chunks)
haiku-rag rebuild --embed-only
```

**Rebuild modes:**

| Mode | Flag | Use case |
|------|------|----------|
| Full | (default) | Changed converter, source files updated |
| Rechunk | `--rechunk` | Changed chunking strategy or chunk size |
| Embed only | `--embed-only` | Changed embedding model or vector dimensions |

### Download Models

Download required runtime models:

```bash
haiku-rag download-models
```

This command downloads:

- Docling OCR/conversion models
- HuggingFace tokenizer (for chunking)
- Ollama models referenced in your configuration (embeddings, QA, research, rerank)

Progress is displayed in real-time with download status and progress bars for Ollama model pulls.

## Time Travel

LanceDB maintains version history for tables, enabling you to query the database as it existed at a previous point in time. This is useful for:

- **Debugging**: Investigate data before a problematic change
- **Auditing**: Verify what knowledge was available when a support ticket was filed

### Query Historical State

Use `--before` to query the database as it existed before a specific datetime:

```bash
# Query documents as of January 15, 2025
haiku-rag --before "2025-01-15" list

# Search historical state
haiku-rag --before "2025-01-15T14:30:00" search "machine learning"

# Ask questions against historical data
haiku-rag --before "2025-01-15" ask "What documents existed?"
```

Supported datetime formats:

- ISO 8601: `2025-01-15T14:30:00`, `2025-01-15T14:30:00Z`, `2025-01-15T14:30:00+00:00`
- Date only: `2025-01-15` (interpreted as start of day)

!!! note
    Time travel mode automatically enables read-only mode. You cannot modify the database while viewing historical state.

### Version History

View version history for database tables:

```bash
# Show history for all tables
haiku-rag history

# Show history for a specific table
haiku-rag history --table documents

# Limit number of versions shown
haiku-rag history --limit 10
```

Output shows version numbers and timestamps, sorted newest first:

```
Version History

documents
  v5: 2025-01-15 14:30:00
  v4: 2025-01-14 10:00:00
  v3: 2025-01-13 09:15:00

chunks
  v8: 2025-01-15 14:30:00
  v7: 2025-01-14 10:00:00
  ...
```

Use the timestamps from `history` to construct `--before` queries.
