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
```

From file or URL:
```bash
haiku-rag add-src /path/to/document.pdf
haiku-rag add-src https://example.com/article.html
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

## Server

Start the MCP server:
```bash
# HTTP transport (default)
haiku-rag serve

# stdio transport
haiku-rag serve --stdio

# SSE transport
haiku-rag serve --sse
```

## Settings

View current configuration settings:
```bash
haiku-rag settings
```

## Maintenance

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
