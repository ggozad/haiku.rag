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

!!! note
    As you add documents to `haiku.rag` the database keeps growing. By default, LanceDB supports versioning
    of your data. Create/update operations are atomic‑feeling: if anything fails during chunking or embedding,
    the database rolls back to the pre‑operation snapshot using LanceDB table versioning. You can optimize and
    compact the database by running the [vacuum](#vacuum-optimize-and-cleanup) command.

### Get Document

```bash
haiku-rag get <TAB>
# or
haiku-rag get 3f4a...   # document ID (autocomplete supported)
```

### Delete Document

```bash
haiku-rag delete <TAB>
haiku-rag rm <TAB>       # alias
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

The QA agent will search your documents for relevant information and provide a comprehensive answer. With `--cite`, responses include citations showing which documents were used.
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

Start the MCP server:
```bash
# HTTP transport (default)
haiku-rag serve

# stdio transport
haiku-rag serve --stdio
```

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

## Migration

### Migrate from SQLite to LanceDB

Migrate an existing SQLite database to LanceDB:

```bash
haiku-rag migrate /path/to/old_database.sqlite
```

This will:
- Read all documents, chunks, embeddings, and settings from the SQLite database
- Create a new LanceDB database with the same data in the same directory
- Optimize the new database for best performance

The original SQLite database remains unchanged, so you can safely migrate without risk of data loss.

## Shell Autocompletion

Enable shell autocompletion for faster, error‑free usage.

- Temporary (current shell only):
  ```bash
  eval "$(haiku-rag --show-completion)"
  ```
- Permanent installation:
  ```bash
  haiku-rag --install-completion
  ```

What’s completed:
- `get` and `delete`/`rm`: Document IDs from the selected database (respects `--db`).
- `add-src`: Local filesystem paths (URLs can still be typed manually).
