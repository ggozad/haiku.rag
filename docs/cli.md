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

### Add Documents

From text:
```bash
haiku-rag add "Your document content here"

# Set a title
haiku-rag add "Your document content here" --title "My Document"

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

From an S3 bucket (requires the `[s3]` extra, see the [ingester docs](ingester.md) for continuous S3 polling):
```bash
# AWS S3 with credentials in the default chain (env vars, IAM role, AWS profile)
haiku-rag add-src s3://my-bucket/path/to/document.pdf

# S3-compatible endpoint (SeaweedFS, MinIO, Cloudflare R2, etc.)
AWS_ACCESS_KEY_ID=key AWS_SECRET_ACCESS_KEY=secret AWS_REGION=us-east-1 \
  AWS_ENDPOINT_URL=http://localhost:8333 \
  haiku-rag add-src s3://my-bucket/path/to/document.pdf
```

!!! note
    When adding a directory, the converter's supported extensions filter applies. For pattern-based ignore/include filtering (e.g. `**/.git/**`), use the [ingester](ingester.md) with a filesystem source.

!!! note
    As you add documents to `haiku.rag` the database keeps growing. By default, LanceDB supports versioning
    of your data. Create/update operations are atomic‑feeling: if anything fails during chunking or embedding,
    the database rolls back to the pre‑operation snapshot using LanceDB table versioning. You can optimize and
    compact the database by running the [vacuum](#vacuum-optimize-and-cleanup) command.

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

### Get Document

```bash
haiku-rag get 3f4a...   # document ID
```

### Delete Document

```bash
haiku-rag delete 3f4a...   # document ID
haiku-rag rm 3f4a...       # alias
```

## Search

Basic search:
```bash
haiku-rag search "machine learning"
```

With options:
```bash
haiku-rag search "python programming" --limit 10  # or -l 10
```

With search type:
```bash
# Hybrid search (the default)
haiku-rag search "python programming" --search-type hybrid  # or -s hybrid

# Full-text search only
haiku-rag search "python programming" --search-type fts  # or -s fts

# Vector search only
haiku-rag search "python programming" --search-type vector  # or -s vector
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

Image-as-query (requires a multimodal embedder):
```bash
haiku-rag search --image path/to/figure.png --limit 5
```

When `--image` is used, the positional query is omitted. Pass one or the other, not both.

## Question Answering

Ask questions about your documents:
```bash
haiku-rag ask "Who is the author of haiku.rag?"
```

Filter to specific documents:
```bash
haiku-rag ask "What are the main findings?" --filter "uri LIKE '%paper%'"
```

`ask` runs the [rag skill](skills/index.md) and always renders citations under the answer. When available, citations use the document title, otherwise they fall back to the URI.

Flags:

- `--filter` / `-f`: Restrict searches to documents matching the filter (see [Filtering Search Results](python.md#filtering-search-results))

## Analyze

Answer complex analytical questions via code execution:

```bash
haiku-rag analyze "How many documents mention security?"
```

Filter to specific documents:

```bash
haiku-rag analyze "What is the total revenue?" --filter "title LIKE '%Financial%'"
```

Flags:

- `--filter` / `-f`: SQL WHERE clause to restrict document access

See [Analysis skill](skills/analysis.md) for details on capabilities and configuration.

## Chat

Launch an interactive chat session for multi-turn conversations:

```bash
haiku-rag chat
haiku-rag chat --db /path/to/database.lancedb

# Enable analysis skill (code execution)
haiku-rag chat -s rag -s analysis
```

!!! note
    Requires the `tui` extra: `pip install haiku.rag-slim[tui]` (included in full `haiku.rag` package)

Flags:

- `--skill` / `-s`: Skills to enable. `rag` (default), `analysis`. Can be repeated for multiple skills.

The chat interface provides:

- Streaming responses with real-time tool execution
- Expandable citations with source metadata
- Session memory for context-aware follow-up questions
- Visual grounding to inspect chunk source locations

See [Chat](chat.md) for keyboard shortcuts and features.

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

See [Tuning: Inspector](tuning.md#inspector) for the full keybindings and modal flows.

## Visualize Chunk

Display visual grounding for a chunk - shows page images with highlighted bounding boxes:

```bash
haiku-rag visualize <chunk_id>
```

This renders the source document pages with the chunk's location highlighted. Useful for verifying chunk boundaries and understanding document structure.

!!! note
    Requires a terminal with image support (iTerm2, Kitty, WezTerm, etc.) and documents processed with docling that have page images stored.

## Database lifecycle

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
- per-table row counts and storage sizes (documents, document_meta, chunks, document_items)
- vector index status (exists/not created, indexed/unindexed chunks)
- table versions per table (documents, document_meta, chunks)

At the end, a separate "Versions" section lists runtime package versions:
- haiku.rag
- lancedb
- docling

### Doctor

Check the database for consistency problems and print a pass/warn/fail report:

```bash
haiku-rag doctor [--db /path/to/your.lancedb]
```

Checks include:

- required tables are present
- `documents` and `document_meta` are in 1:1 correspondence
- chunks and document items reference documents that exist
- every document produced chunks and document items
- chunk `doc_item_refs` resolve to existing document items
- chunk vector size matches the stored embedding dimension
- chunks are embedded (no all-zero vectors)
- picture items carry their image data
- exactly one settings row is present
- the configured embedding identity matches the stored settings
- no database migrations are pending
- the vector index covers all chunks
- API keys are set for configured providers

Each failure prints the command that fixes it (`rebuild`, `create-index`, `migrate`, `rebuild --set-embedder`). `doctor` makes no changes. It exits with status 1 when any check fails, so it can gate CI or monitoring.

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
Applied 4 migration(s):
  - 0.20.0: Add 'docling_document_json' and 'docling_version' columns
  - 0.23.1: Add content_fts column for contextualized FTS search
  - 0.25.0: Compress docling_document with gzip
  - 0.38.0: Split docling_document pages into separate column and re-compress with zstd
Migration completed successfully.
```

!!! tip
    Back up your database before running migrations. While migrations are designed to be safe, having a backup provides peace of mind for production databases.

### Download Models

Download required runtime models:

```bash
haiku-rag download-models
```

This command downloads:

- Docling OCR/conversion models
- HuggingFace tokenizer (for chunking)
- Ollama models referenced in your configuration (embeddings, QA, rerank)

Progress is displayed in real-time with download status and progress bars for Ollama model pulls.

## Maintenance

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

### Rebuild Database

Rebuild the database by re-indexing documents. Useful when switching embeddings provider/model or changing chunking settings:

```bash
# Full rebuild (default) - re-converts from source files, re-chunks, re-embeds
haiku-rag rebuild

# Re-chunk from stored content (no source file access)
haiku-rag rebuild --rechunk

# Only regenerate embeddings (fastest, keeps existing chunks)
haiku-rag rebuild --embed-only

# Only generate titles for untitled documents
haiku-rag rebuild --title-only

# Run the VLM over already-stored picture bytes and patch descriptions
# into the docling blob. Skips the docling parse entirely.
haiku-rag rebuild --descriptions

# Adopt the current embedder identity without re-embedding (same vector dimension)
haiku-rag rebuild --set-embedder
```

**Rebuild modes:**

| Mode | Flag | Use case |
|------|------|----------|
| Full | (default) | Changed converter, source files updated |
| Rechunk | `--rechunk` | Changed chunking strategy or chunk size |
| Embed only | `--embed-only` | Changed embedding model or vector dimensions |
| Title only | `--title-only` | Generate titles for documents without one |
| Descriptions | `--descriptions` | Add VLM picture descriptions to an existing database |
| Set embedder | `--set-embedder` | Same model, different serving stack (e.g. Ollama to vLLM); vector dimension unchanged |

**`--set-embedder` mode** updates the stored embedding provider/name to match the current config without re-embedding, valid only when the vector dimension is unchanged. Use it when the same model is served by a different stack so the recorded identity stops drifting from the config. A changed vector dimension is rejected; regenerate embeddings with `--embed-only` or a full rebuild instead.

**`--descriptions` mode** runs the configured VLM (`processing.conversion_options.picture_description.model`) over the picture bytes already stored in `document_items.picture_data`, patches each description into the stored docling blob's `pictures[i].meta.description.text`, and re-chunks + re-embeds so chunk text reflects the new descriptions. Requires `processing.pictures: description` in the config. Idempotent: pictures that already carry a description are skipped, so the operation is safe to re-run after a partial failure. The docling parse is skipped entirely. Only the VLM time is paid.

### Vacuum (Optimize and Cleanup)

Reduce disk usage by optimizing and pruning old table versions across all tables:

```bash
haiku-rag vacuum
```

**Automatic Cleanup:** Vacuum runs automatically in the background after document operations, throttled to at most once every 5 minutes so sustained ingestion does not trigger continuous compaction (a final vacuum runs when the client closes). By default, it removes versions older than 1 day (configurable via `storage.vacuum_retention_seconds`), preserving recent versions for concurrent connections. Manual vacuum can be useful for cleanup after bulk operations or to free disk space immediately.

## MCP Server

```bash
# HTTP transport on port 8001
haiku-rag mcp

# stdio transport (for Claude Desktop)
haiku-rag mcp --stdio

# Custom port
haiku-rag mcp --port 9000

# Bind to all interfaces (containers, trusted LAN)
haiku-rag mcp --host 0.0.0.0

# Read-only mode (no write tools)
haiku-rag --read-only mcp
```

See [MCP](mcp.md) for details. For continuous document ingestion
(filesystem watch, S3 polling, HTTP / WebDAV sources), use the
[ingester](ingester.md).

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

## Create Skill

Generate a standalone skill package with an embedded database:

```bash
haiku-rag create-skill --name myskill --db /path/to/database.lancedb
```

The generated package is a pip-installable Python package that registers as a `haiku.skills` entry point.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Skill name (lowercase alphanumeric and hyphens, required) | — |
| `--db` | Path to LanceDB database to embed (required) | — |
| `--description` | Skill description | Standard RAG description |
| `--tools` | Comma-separated tool names, or `all` | `all` |
| `--preamble` | Custom preamble for skill instructions | Standard RAG preamble |
| `--config-file` | Path to `haiku.rag.yaml` to embed | None |
| `--output` / `-o` | Output directory | Current directory |

### Available Tools

`cite`, `execute_code`, `get_document`, `list_documents`, `search`

### Example

```bash
# Generate a skill with specific tools and custom preamble
haiku-rag create-skill \
  --name medic \
  --db /path/to/medic.lancedb \
  --tools search,cite \
  --config-file /path/to/haiku.rag.yaml \
  --description "Military medic knowledge base" \
  --preamble "You are a military medic expert."

# Install the generated package
uv pip install -e ./medic-skill

# Use with haiku-skills
haiku-skills chat --use-entrypoints --skill medic
```

### Generated Package Structure

```
{name}-skill/
├── pyproject.toml
└── {name}_skill/
    ├── __init__.py    # create_skill() entry point
    ├── SKILL.md       # Skill metadata and instructions
    └── assets/
        ├── {name}.lancedb/   # Embedded database
        └── haiku.rag.yaml    # Optional config
```

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
