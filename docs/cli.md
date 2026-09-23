# Command line interface

`haiku-rag` manages documents, searches and answers from the command line, and runs the MCP server. `haiku-rag COMMAND -h` shows a command's options.

## Global options

Global options precede the command:

| Option | Meaning |
|---|---|
| `--config PATH` | Configuration file. `HAIKU_RAG_CONFIG_PATH` does the same |
| `--db-name NAME` | Work on one database from `lancedb.databases` |
| `--read-only` | Open the database read-only. `search`, `ask`, `list`, `get`, `visualize`, `chat` and `inspect` always do |
| `--version`, `-v` | Show the version and exit |

`--db PATH` follows the command and opens the database at that path, named by its stem, whatever the configuration places. `settings`, `init-config` and `download-models` take no `--db`.

```bash
haiku-rag --config /path/to/config.yaml list --db /path/to/custom.lancedb
haiku-rag --db-name papers list
```

With several databases configured, `search`, `ask`, `chat` and `mcp` cover them all, and every other command needs `--db-name` or `--db`. See [Multiple databases](configuration/multiple-databases.md#cli).

## Documents

### Add

From text:

```bash
haiku-rag add "Your document content here" --title "My Document" --meta author=alice --meta topic=notes
```

From a file, URL, directory or S3 object:

```bash
haiku-rag add-src /path/to/document.pdf --title "Q3 Financial Report"
haiku-rag add-src https://example.com/article.html
haiku-rag add-src /path/to/documents/            # recursive
haiku-rag add-src s3://my-bucket/path/to/document.pdf
```

`--meta KEY=VALUE` is repeatable. Values are parsed as JSON where possible (numbers, booleans, null, arrays, objects) and kept as strings otherwise.

A directory is walked recursively. Files are kept when docling-local or the text handler supports their extension, and `--title` is ignored. For include and ignore patterns, use the [ingester](ingester.md) with a filesystem source.

`s3://` needs the `s3` extra. Credentials come from the default AWS chain. For an S3-compatible store, set `AWS_ENDPOINT_URL`:

```bash
AWS_ACCESS_KEY_ID=key AWS_SECRET_ACCESS_KEY=secret AWS_REGION=us-east-1 \
  AWS_ENDPOINT_URL=http://localhost:8333 \
  haiku-rag add-src s3://my-bucket/path/to/document.pdf
```

A failed add or update rolls the database back to its state before the operation.

### List, get, delete

```bash
haiku-rag list
haiku-rag list -f "uri LIKE '%arxiv%'"
haiku-rag list -f "uri LIKE '%.pdf' AND title LIKE '%paper%'"

haiku-rag get 3f4a...      # document ID
haiku-rag delete 3f4a...   # or: haiku-rag rm 3f4a...
```

`--filter`/`-f` takes a SQL WHERE clause over the document columns, see [Filtering search results](python.md#filtering-search-results).

## Search and ask

```bash
haiku-rag search "machine learning"
haiku-rag search "python programming" -l 10 -s fts -f "uri LIKE '%arxiv%'"
haiku-rag search --image path/to/figure.png
```

| Option | Meaning |
|---|---|
| `--limit`, `-l` | Results to return. Default `search.limit` |
| `--search-type`, `-s` | `hybrid` (default), `vector` or `fts` |
| `--filter`, `-f` | SQL WHERE clause over document columns |
| `--image PATH` | Search by image, in place of the text query. Needs a multimodal embedder, and takes no `--search-type` |

`search` prints the matched chunks without context expansion.

```bash
haiku-rag ask "What are the main findings?" -f "uri LIKE '%paper%'"
haiku-rag ask "Does this photo satisfy the spec in the design document?" --image photo.jpg
haiku-rag ask "What are the main findings?" --full-citations
```

`ask` runs the [RAG capability](capabilities/rag.md) and prints citations under the answer, labelled `title (uri)` or whichever of the two the document has. Citation text is cut to a 300-character preview unless `--full-citations` is set. `--image` is repeatable. Retrieval stays text-based, and the QA model needs `vision: true`.

## Terminal apps

| Command | What it does |
|---|---|
| `haiku-rag chat [--model PROVIDER:NAME]` | Conversational RAG in the terminal. See [Chat](chat.md) |
| `haiku-rag inspect` | Browse documents and chunks, search, preview expanded context. See [Inspector](chat.md#inspector) |
| `haiku-rag visualize CHUNK_ID [--no-expand]` | Draw a chunk on its page images, its expanded context fainter. `--no-expand` draws the chunk alone |

`chat` and `inspect` need the `tui` extra, which the full package includes. `visualize` needs a terminal with inline images (iTerm2, Kitty, WezTerm) and documents converted with page images.

## Database lifecycle

### init

```bash
haiku-rag init [--db /path/to/your.lancedb]
```

Creates the database with the configured settings. On an existing database it warns and exits 0. Commands that read or write documents fail when the database does not exist. `info` and `history` report the missing path and exit 0.

### info

```bash
haiku-rag info [--db /path/to/your.lancedb]
```

Shows the database path, the stored haiku.rag version, the embedding provider, model and dimension, per-table row counts and sizes, vector index status, table versions, and pending migrations. A final section lists the haiku.rag, lancedb, docling, pydantic-ai and DoclingDocument schema versions.

### doctor

```bash
haiku-rag doctor [--db /path/to/your.lancedb] [--duplicates-out groups.yaml]
```

Checks the database and prints a pass, warn or fail report. It makes no changes, prints the command that fixes each failure (`rebuild`, `create-index`, `vacuum`, `migrate`, `rebuild --set-embedder`), and exits 1 when any check fails or the database is missing. The checks:

- required tables are present, and `documents` and `document_meta` correspond one to one
- chunks and document items reference documents that exist
- documents with text produced chunks. Empty and heading- or furniture-only documents are not flagged, and image-only documents are flagged according to whether the embedder indexes images
- chunked documents have document items, and chunk `doc_item_refs` resolve to them
- chunk vectors have the stored dimension and are not all zero
- pictures in image and PDF documents carry their image data
- exactly one settings row exists, and the configured embedder matches it
- no migrations are pending
- the vector index covers all chunks, and the full-text index covers the chunks it searches
- near-identical documents, by embedding-centroid similarity. Advisory only, tuned by `doctor.duplicates`. The largest member of each group is suggested to keep
- API keys are set for the configured providers

It also probes the endpoints the configuration uses: Ollama and its models (`{base_url}/api/tags`), docling-serve when configured (`{base_url}/health`), and custom OpenAI-compatible and vLLM endpoints (`{base_url}/models`). Hosted providers get only the API-key check. In-process models have no endpoint and are reported as such.

`--duplicates-out PATH` writes the near-duplicate groups to YAML: per group, `keep` and a list of `documents`, each with `document_id`, `document`, `chunks`, `similarity` and `keep_suggested`.

### migrate

```bash
haiku-rag migrate [--db /path/to/your.lancedb]
```

A database written by an older haiku.rag may need a schema migration. Opening it then fails:

```
Error: Database requires migration from 0.19.0 to 0.38.0. 4 migration(s) pending. Run 'haiku-rag migrate' to upgrade.
```

`migrate` applies the pending migrations and lists them:

```
Applied 4 migration(s):
  - 0.20.0: Add 'docling_document_json' and 'docling_version' columns to documents table
  - 0.23.1: Add content_fts column for contextualized FTS search
  - 0.25.0: Compress docling_document and use large_binary type
  - 0.38.0: Split docling_document pages into separate column and re-compress with zstd
Migration completed successfully.
```

Back up the database first. `migrate` exits 1 on failure.

### download-models

```bash
haiku-rag download-models
```

Fetches the models the configuration needs, see [Installation](installation.md#pre-download-models). Exits 1 on error.

## Maintenance

### rebuild

```bash
haiku-rag rebuild [--rechunk | --embed-only | --title-only | --descriptions | --set-embedder]
```

| Mode | Flag | Use it when |
|------|------|----------|
| Full | (default) | The converter or conversion options changed, or source files were updated. Re-converts from source, re-chunks, re-embeds |
| Rechunk | `--rechunk` | Chunking settings changed. Re-chunks stored content, re-embeds |
| Embed only | `--embed-only` | The embedding model or `vector_dim` changed. Keeps chunks |
| Title only | `--title-only` | Documents lack titles |
| Descriptions | `--descriptions` | Adding VLM picture descriptions to an existing database |
| Set embedder | `--set-embedder` | The same model is now served by another stack (e.g. Ollama to vLLM) |

`--set-embedder` records the configured embedding provider and name without re-embedding, and is rejected when the vector dimension changed.

`--descriptions` runs `processing.conversion_options.picture_description.model` over the picture bytes already stored, writes each description into the stored docling document, then re-chunks and re-embeds. It needs `processing.pictures: description`, skips docling conversion, and skips pictures that already have a description, so it is safe to re-run.

### vacuum

```bash
haiku-rag vacuum
```

Compacts the tables and removes old versions. With `storage.auto_vacuum` it also runs in the background, see [Storage](configuration/storage.md#local-storage).

### create-index

```bash
haiku-rag create-index [--db /path/to/your.lancedb]
```

Builds an IVF_PQ vector index over the chunks, using `search.vector_index_metric`. It needs at least 256 chunks. Without an index, search is exact brute-force kNN, which is fast enough below about 100,000 chunks. Re-run it after substantial growth to retrain the centroids. See [Vector indexing](configuration/storage.md#vector-indexing).

## Tags and history

A tag names the current database state: one LanceDB tag on each of the five tables, taken from a single version snapshot.

```bash
haiku-rag tag create release-1   # e.g. at deploy time
haiku-rag tag list               # tags and the versions they point to
haiku-rag tag delete release-1   # release its versions for cleanup
haiku-rag tag restore release-1
```

A tag missing from some tables, created outside haiku.rag or left by a failure, is partial. `tag list` marks it, and it can be deleted but never restored.

Create and restore tags with every other writer stopped. The snapshot is coordinated within one process, so a writer in another process can commit between the per-table reads.

Vacuum keeps the oldest tagged version and everything newer. Delete tags you no longer need so cleanup can advance.

`tag restore` changes the live state: each table gets a new latest version equal to the tagged one. Before changing anything it creates a safety tag, `before-restore-<timestamp>`, and reports it:

```bash
haiku-rag tag restore release-1 --db /path/to/db.lancedb --yes
haiku-rag tag restore before-restore-YYYYMMDDTHHMMSSZ --db /path/to/db.lancedb --yes
```

Restore is coordinated but not atomic across tables. On failure it attempts to roll back to the pre-restore state and reports whether that succeeded. `--yes` only skips the confirmation prompt. Restore never migrates: restoring a tag from an older version succeeds, and the next open reports the migration to run. Tag commands exit 1 on failure.

```bash
haiku-rag history                       # every table
haiku-rag history -t documents -l 10    # one table, 10 versions
```

`history` lists table versions newest first, with tags marked. `--table` takes `documents`, `document_meta`, `chunks`, `document_items` or `settings`:

```
documents
  v5: 2025-01-15 14:30:00  <- release-1
  v4: 2025-01-14 10:00:00
```

## Servers

```bash
haiku-rag mcp                  # streamable HTTP on 127.0.0.1:8001
haiku-rag mcp --stdio          # stdio, for Claude Desktop
haiku-rag mcp --host 0.0.0.0 --port 9000
```

See [MCP](mcp.md). Continuous ingestion runs in the separate [`haiku-ingester`](ingester.md) service.

## Settings

```bash
haiku-rag settings               # the effective configuration
haiku-rag init-config [PATH]     # write every setting with its default, ./haiku.rag.yaml by default
```

`init-config` refuses to overwrite an existing file and exits 1.
