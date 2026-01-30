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
# Filter by URI pattern (--filter or -f)
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
haiku-rag search "python programming" --limit 10  # or -l 10
```

With filters (filter by document properties, use `--filter` or `-f`):
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

Provide background context for the question:
```bash
haiku-rag ask "What are the protocols?" --context "Focus on security best practices"
haiku-rag ask "Summarize the findings" --context-file background.txt
```

The QA agent searches your documents for relevant information and provides a comprehensive answer. When available, citations use the document title; otherwise they fall back to the URI.

Flags:

- `--cite`: Include citations showing which documents were used
- `--deep`: Decompose the question into sub-questions answered in parallel before synthesizing a final answer
- `--filter` / `-f`: Restrict searches to documents matching the filter (see [Filtering Search Results](python.md#filtering-search-results))
- `--context`: Background context for the question (passed to the agent as system context)
- `--context-file`: Path to a file containing background context

## Chat

Launch an interactive chat session for multi-turn conversations:

```bash
haiku-rag chat
haiku-rag chat --db /path/to/database.lancedb
```

Provide initial background context for the conversation:
```bash
haiku-rag chat --initial-context "Focus on Python programming concepts"
```

!!! note
    Requires the `tui` extra: `pip install haiku.rag-slim[tui]` (included in full `haiku.rag` package)

The chat interface provides:

- Streaming responses with real-time tool execution
- Expandable citations with source metadata
- Session memory for context-aware follow-up questions
- Visual grounding to inspect chunk source locations
- Initial context that can be edited before the first message

**Initial Context Behavior:**

- Edit initial context via command palette before sending your first message
- Once you send a message, initial context becomes read-only
- The agent uses initial context as a starting point for session summarization
- Clearing chat resets to the CLI-provided context and unlocks editing

Flags:

- `--initial-context`: Initial background context for the conversation (editable until first message)

See [Applications](apps.md#chat-tui) for keyboard shortcuts and features.

## Inspect

Launch the interactive inspector TUI for browsing documents and chunks:

```bash
haiku-rag inspect
haiku-rag inspect --db /path/to/database.lancedb
```

!!! note
    Requires the `tui` extra: `pip install haiku.rag-slim[tui]` (included in full `haiku.rag` package)

The inspector provides:

- Browse all documents in the database
- View document metadata and content
- Explore individual chunks
- Search and filter results

See [Applications](apps.md#inspector) for details.

## Research

Run the multi-step research graph:

```bash
haiku-rag research "How does haiku.rag organize and query documents?"
```

Filter to specific documents:

```bash
haiku-rag research "What are the key findings?" --filter "uri LIKE '%paper%'"
```

Provide background context for the research:

```bash
haiku-rag research "What are the safety protocols?" --context "Industrial manufacturing context"
haiku-rag research "Analyze the methodology" --context-file research-background.txt
```

Flags:

- `--filter` / `-f`: SQL WHERE clause to filter documents (see [Filtering Search Results](python.md#filtering-search-results))
- `--context`: Background context for the research
- `--context-file`: Path to a file containing background context

Research parameters like `max_iterations` and `max_concurrency` are configured in your [configuration file](configuration/index.md) under the `research` section.

## RLM (Recursive Language Model)

Answer complex analytical questions via code execution:

```bash
haiku-rag rlm "How many documents mention security?"
```

Filter to specific documents:

```bash
haiku-rag rlm "What is the total revenue?" --filter "title LIKE '%Financial%'"
```

Pre-load specific documents for comparison:

```bash
haiku-rag rlm "Compare the conclusions" --document "Report A" --document "Report B"
```

Flags:

- `--filter` / `-f`: SQL WHERE clause to restrict document access
- `--document` / `-d`: Pre-load a document by title or ID (can repeat)

See [RLM Agent](rlm.md) for details on capabilities and configuration.

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

### Generate Configuration File

Generate a YAML configuration file with defaults:
```bash
haiku-rag init-config [output_path]
```

If no path is specified, creates `haiku.rag.yaml` in the current directory.

## Database Management

### Initialize Database

Create a new database:

```bash
haiku-rag init [--db /path/to/your.lancedb]
```

This creates the database with the configured settings. **All other commands require an existing database** - they will fail with an informative error if the database doesn't exist.

### Migrate Database

Apply pending database migrations:

```bash
haiku-rag migrate [--db /path/to/your.lancedb]
```

When you upgrade haiku.rag to a new version that includes schema changes, the database requires migration. Opening a database with pending migrations will display an error:

```
Error: Database requires migration from 0.19.0 to 0.26.5. 3 migration(s) pending. Run 'haiku-rag migrate' to upgrade.
```

Run `haiku-rag migrate` to apply the pending migrations. The command shows which migrations were applied:

```
Applied 3 migration(s):
  - 0.20.0: Add 'docling_document_json' and 'docling_version' columns
  - 0.23.1: Add content_fts column for contextualized FTS search
  - 0.25.0: Compress docling_document with gzip
Migration completed successfully.
```

!!! tip
    Back up your database before running migrations. While migrations are designed to be safe, having a backup provides peace of mind for production databases.

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
