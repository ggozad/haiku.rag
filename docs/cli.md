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

A command started with `TRACEPARENT` / `TRACESTATE` in its environment joins that trace. `mcp`, `chat` and `inspect` never join. The parent's sampling decision applies, so a `TRACEPARENT` with the sampled flag `00` suppresses the command's spans.

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
haiku-rag doctor [--db /path/to/your.lancedb] [--duplicates-out groups.yaml] [--skip-providers] [--json]
```

Checks the database and prints a pass, warn or fail report. It makes no changes, prints the command that fixes each failure (`rebuild`, `create-index`, `vacuum`, `migrate`, `rebuild --embed-only`, `rebuild --reindex`), and exits 1 when any check fails or the database is missing. The checks:

- required tables are present, and `documents` and `document_meta` correspond one to one
- chunks and document items reference documents that exist
- documents with text produced chunks. Empty and heading- or furniture-only documents are not flagged, and image-only documents are flagged according to whether the embedder indexes images
- chunked documents have document items, and chunk `doc_item_refs` resolve to them
- chunk vectors have the stored dimension and are not all zero
- pictures in image and PDF documents carry their image data
- exactly one settings row exists, and the configured embedder matches it
- no migrations are pending
- the vector index covers all chunks, and the full-text index exists and covers rows. Rows written since it was last built are a warning
- near-identical documents, by embedding-centroid similarity. Advisory only, tuned by `doctor.duplicates`. The largest member of each group is suggested to keep
- API keys are set for the configured providers

These are warnings and do not change the exit code: rows outside the full-text or vector index, no vector index on a large table, near-duplicate documents, picture-only documents without chunks, pictures without image data, an embedder that differs from the recorded one at the same vector dimension, and an Ollama endpoint missing a configured model. Every other problem fails.

It also probes the endpoints the configuration uses: Ollama and its models (`{base_url}/api/tags`), docling-serve when configured (`{base_url}/health`), and custom OpenAI-compatible and vLLM endpoints (`{base_url}/models`). Hosted providers get only the API-key check. In-process models have no endpoint and are reported as such. `--skip-providers` skips the API-key check and the endpoint probes.

`--json` prints the report as JSON on stdout and nothing else, with the same exit code: `{"results": [...]}`, each result carrying `name`, `severity` (`ok`, `warn` or `fail`), `message`, `remediation` and `details`. A missing database is one `database_missing` failure.

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
haiku-rag rebuild [--rechunk | --embed-only | --title-only | --descriptions | --set-embedder | --reindex]
```

| Mode | Flag | Use it when |
|------|------|----------|
| Full | (default) | The converter or conversion options changed, or source files were updated. Re-converts from source, re-chunks, re-embeds |
| Rechunk | `--rechunk` | Chunking settings changed, or an upgrade changed how chunks or document items are extracted. Re-chunks stored content, re-embeds, recreates document items |
| Embed only | `--embed-only` | The embedding model or `vector_dim` changed. Keeps chunks |
| Title only | `--title-only` | Documents lack titles |
| Descriptions | `--descriptions` | Adding VLM picture descriptions to an existing database |
| Set embedder | `--set-embedder` | The same model is now served by another stack (e.g. Ollama to vLLM) |
| Reindex | `--reindex` | `doctor` reports rows outside the full-text index, or an index is missing |

Every mode except `--title-only` and `--set-embedder` ends by rebuilding the full-text and scalar indexes, whatever `storage.auto_vacuum` says. A full rebuild, `--rechunk`, `--embed-only` and `--descriptions` also retrain the vector index if the database had one. `--reindex` rebuilds the full-text and scalar indexes from scratch and nothing else: it rewrites no rows and leaves the vector index alone.

A full rebuild, `--rechunk` and `--descriptions` replace the chunks of 50 documents at a time, each batch in one transaction, and leave the other documents' chunks as they were. A run that fails or is cancelled rolls its batch back and keeps every document searchable and the vector index in place. Run it again to finish. A process killed outright, by SIGKILL or the out-of-memory killer, can leave the batch it was writing without chunks until you run the rebuild again. When the vector dimension changed, they drop the chunks table before processing any document, so an interrupted run leaves the documents it did not reach without chunks.

The database records the embedder whose vectors the chunks table holds. A rebuild that keeps the table records the configured embedder once it has finished. An interrupted one to another embedder at the same dimension keeps the old one recorded while some documents already have the new one's vectors, so opening the database fails when writable and warns when read-only, until you run the rebuild again. A rebuild that empties the table, `--embed-only` or a dimension change, records the configured embedder as soon as the table is recreated. `--title-only` embeds nothing and records nothing.

`--set-embedder` records the configured embedding provider and name without re-embedding, and is rejected when the vector dimension changed.

`--descriptions` runs `processing.conversion_options.picture_description.model` over the picture bytes already stored, writes each description into the stored docling document, then re-chunks and re-embeds. It needs `processing.pictures: description`, skips docling conversion, and skips pictures that already have a description, so it is safe to re-run.

### vacuum

```bash
haiku-rag vacuum [--retention-seconds N]
```

Compacts the tables and removes old versions older than `storage.vacuum_retention_seconds` (default a day). `--retention-seconds` overrides it for one run. `--retention-seconds 0` removes every version except the current one and those kept by a tag, so stop every other process using the database first. With `storage.auto_vacuum` vacuum also runs in the background, see [Storage](configuration/storage.md#local-storage).

### create-index

```bash
haiku-rag create-index [--db /path/to/your.lancedb]
```

Builds an IVF_PQ vector index over the chunks, using `search.vector_index_metric`. It needs at least 256 chunks. Without an index, search is exact brute-force kNN, which is fast enough below about 100,000 chunks. Re-run it after substantial growth to retrain the centroids. A `rebuild` that rewrites chunks retrains an existing index and never creates one. See [Vector indexing](configuration/storage.md#vector-indexing).

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

Vacuum keeps every tagged version and the files it references, and removes the untagged versions older than the retention. Delete tags you no longer need so their files can be removed.

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
