# Database and Storage

## Operational constraints

Four things to know before deploying.

**Run one writer per database.** This is a haiku.rag constraint, not a LanceDB
one. A write that spans several tables is serialized by an in-process lock and
rolled back by restoring each table to the version it had when the write started.
Both are process-local: a second writing process can commit between that snapshot
and the mutation, and a rollback would then revert its work along with ours. Run
a single writer, either the [`haiku-ingester`](../ingester.md) service or your own
application. Read-only consumers are unrestricted.

**Readers lag by an interval.** A connection always sees its own writes. It sees
another process's writes after `lancedb.read_consistency_interval_seconds`
(default 30).

**Migrate after an upgrade that changes the schema.** `haiku-rag migrate` applies
pending migrations in place, and `haiku-rag info` lists what is pending. A
release that needs it says so in the [changelog](../changelog.md).

**The embedding dimension is fixed per database.** Every chunk vector has the
dimension the database was created with. Changing `embeddings.model.vector_dim`
raises `ConfigMismatchError` on open, because stored vectors cannot be compared
against new ones. Changing the provider or model name while keeping the dimension
warns on a read-only open and raises on a writable one. `haiku-rag rebuild
--set-embedder` adopts the new identity without re-embedding, and `haiku-rag
rebuild --embed-only` re-embeds against the new model.

## Local Storage

By default, `haiku.rag` uses a local LanceDB database:

```yaml
storage:
  data_dir: /path/to/data  # Empty = use default platform location
  auto_vacuum: true  # Enable automatic vacuuming after operations
  vacuum_retention_seconds: 86400  # Cleanup threshold in seconds
```

- **data_dir**: Directory for local database storage. When empty, uses platform-specific default locations
- **auto_vacuum**: When enabled (default), automatically runs vacuum after document create/update/delete operations and database rebuilds. Background vacuums are throttled to at most one every 5 minutes, so sustained ingestion does not trigger continuous compaction, and a final vacuum runs when the client closes. Set to `false` to disable automatic vacuuming and rely on manual `haiku-rag vacuum` commands only. Disabling can help avoid potential crashes in high-concurrency scenarios
- **vacuum_retention_seconds**: When vacuum runs, old table versions older than this threshold are removed. Default: 86400 seconds (1 day). Set to 0 for aggressive cleanup (removes all old versions immediately)

!!! warning "Vacuum Retention Threshold"
    The `vacuum_retention_seconds` value should be larger than the typical time it takes to process and write a document. If a concurrent operation is in progress while vacuum runs, setting this value too low can cause race conditions where vacuum removes table versions that an in-flight operation still needs. The default of 86400 seconds (1 day) is conservative and safe for most use cases.

### Vacuum Memory Requirements

Vacuum compacts small data files into larger ones. LanceDB targets roughly one million rows per fragment, which a `documents` table holding multi-megabyte docling blobs never reaches, so each vacuum that follows new documents re-merges the whole existing fragment rather than only the new ones. Peak memory therefore scales with the total size of the `documents` table, not with how much was added.

Measured peak resident memory is about 5x the size of the `documents` table's data files. An 8.8 GB table peaked at 48.7 GB. Plan for **6x the size of `documents/` on disk** as available RAM, or the vacuum will be killed by the OOM killer partway through.

Check the current size with:

```bash
du -sh /path/to/database.lancedb/documents.lance
```

If that number times six exceeds available RAM, use one of:

- Reduce `images_scale` (see [Image Settings](processing.md#image-settings)). Rendered page rasters dominate the size of `documents`, and their byte cost falls with the square of the scale factor.
- Set `generate_page_images: false` if visual grounding through `visualize_chunk()` is not needed. This removes page rasters entirely.
- Set `auto_vacuum: false` and run `haiku-rag vacuum` manually when the machine is otherwise idle, so the peak does not land alongside ingestion.

Vacuum also folds new rows into the full-text index. Search stays correct without it but scans the uncovered rows on every query. `haiku-rag doctor` reports the coverage.

This is an upstream limitation rather than a `haiku.rag` setting. Compaction bounds itself by row count instead of bytes, and LanceDB's async API exposes no batch size or fragment target to override it. Tracked at [lancedb/lancedb#2325](https://github.com/lancedb/lancedb/issues/2325). The requirement above will drop once compaction batches by bytes.

### Placing the Database

`lancedb.databases` maps a name to a location, a local path or a URI, and is the one way to place databases. With nothing configured, the database is the entry `haiku.rag` at `<storage.data_dir>/haiku.rag.lancedb`. To put one database somewhere else, name it:

```yaml
lancedb:
  databases:
    notes: /data/notes.lancedb
```

The name is what `source` carries in search results, citations and documents, and what `--db-name` and `sources` select. The default database answers to `haiku.rag`.

An explicit `--db PATH` on the command line opens that database instead, named by the path's stem, whatever is configured. From Python, `db_path` places the database only where the configuration places none: beside `lancedb.databases` it raises `AmbiguousDatabaseError`.

A value with no scheme is a local path wherever it is configured, so `haiku-rag init` creates it and every command that opens an existing database requires it to exist. A mistyped path fails rather than becoming a new empty database.

## Database Creation

Databases must be explicitly created before use:

**CLI:**
```bash
# Create in default location (see Configuration File Locations below)
haiku-rag init

# Create at custom path
haiku-rag init --db /path/to/database.lancedb
```

**Python:**
```python
# Create at custom path
async with HaikuRAG("/path/to/database.lancedb", create=True) as client:
    ...

# Create in default location
async with HaikuRAG(create=True) as client:
    ...
```

The [default location](index.md#configuration-file-locations) is platform-specific (e.g., `~/Library/Application Support/haiku.rag/` on macOS).

Opening a nonexistent local database given as a path raises `FileNotFoundError`, naming the path. This prevents accidental database creation from typos or misconfigured paths. A configured or default database raises `SourceUnavailableError` instead, naming the database and not its location.

## Remote Storage

For remote storage, give the database a URI as its location. Credentials and storage options are connection settings, shared by every database in the configuration:

```yaml
# LanceDB Cloud
lancedb:
  databases:
    papers: db://your-database-name
  api_key: your-api-key
  region: us-west-2

# Amazon S3
lancedb:
  databases:
    papers: s3://my-bucket/my-table
  storage_options:
    region: us-east-1

# Amazon S3 with explicit credentials
lancedb:
  databases:
    papers: s3://my-bucket/my-table
  storage_options:
    aws_access_key_id: YOUR_ACCESS_KEY
    aws_secret_access_key: YOUR_SECRET_KEY
    region: us-east-1

# S3-compatible (SeaweedFS, Tigris, etc.)
lancedb:
  databases:
    papers: s3://my-bucket/my-table
  storage_options:
    endpoint: http://localhost:8333
    aws_access_key_id: YOUR_ACCESS_KEY
    aws_secret_access_key: YOUR_SECRET_KEY
    region: us-east-1
    allow_http: "true"

# Azure Blob Storage
lancedb:
  databases:
    papers: az://my-container/my-table

# Google Cloud Storage
lancedb:
  databases:
    papers: gs://my-bucket/my-table

# HDFS
lancedb:
  databases:
    papers: hdfs://namenode:port/path/to/table
```

- **LanceDB Cloud** (`db://`): Requires `api_key` and `region`. Table optimization and indexing are managed server-side.
- **Object storage** (`s3://`, `gs://`, `az://`, `hdfs://`): Uses `storage_options` for credentials and endpoint configuration. Authentication can also be provided via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.) or cloud provider SDK defaults (AWS CLI, Azure CLI, gcloud).
- **S3-compatible stores** (MinIO, Tigris, etc.): Set `endpoint` in `storage_options`. When using `http://` endpoints, also set `allow_http: "true"`.
- **Local path** (no scheme): a location without a scheme is a local path. See [Placing the Database](#placing-the-database).

The `storage_options` keys are case-insensitive and passed directly to the underlying object store library. Available keys depend on the backend. See the [LanceDB storage docs](https://lancedb.com/docs/storage/) for details.

**Note:** Table optimization is automatically handled by LanceDB Cloud (`db://` URIs) and is disabled for better performance. For object storage backends (S3, Azure, GCS), optimization and vector indexing are still performed normally.

### Caching and Read Consistency

```yaml
lancedb:
  read_consistency_interval_seconds: 30   # null to never re-check
  index_cache_size_bytes: 536870912       # null for the LanceDB default
  metadata_cache_size_bytes: 268435456
```

- **read_consistency_interval_seconds**: how often a connection checks for writes from another process. `null` never checks, so a long-lived reader never sees the ingester's writes. `0` checks on every read.
- **index_cache_size_bytes** / **metadata_cache_size_bytes**: sizes for the caches held by the LanceDB session, which is shared across every connection in the process. The first vector query loads the index into it, so on object storage the cache is what stops the next connection refetching it. Size it for the total set of indexes a process keeps warm, against the memory available to it.

### Deployment Pattern: One Writer, Many Readers

The [one-writer constraint](#operational-constraints) shapes the deployment: one
writing process per database URI, any number of read-only consumers.

The recommended layout for production is "different buckets, same account, separate IAM roles per process":

- **Ingestion process** — IAM role with `s3:Get/List` on the documents bucket and `s3:Get/Put/Delete` on the LanceDB bucket. Runs `haiku-ingester serve` (with `ingester.sources[type=s3]` pointing at the documents bucket). Exactly one such process per LanceDB URI.
- **Consumer processes** (1..N) — IAM role with `s3:Get/List` on the LanceDB bucket only. Run `haiku-rag mcp`, the chat TUI, etc. They never see the documents bucket.

Each process picks up its own credentials from the AWS default chain (env vars, IAM instance role, AWS profile), so no credentials are hard-coded in the configuration files.

`haiku-ingester` writes the database the configuration places, so a one-entry `lancedb.databases` needs no further option. `--db PATH` overrides it. When `lancedb.databases` contains more than one database the ingester has no way to name which it writes, and refuses to start with `AmbiguousDatabaseError`: give each database its own ingester process, each with a configuration naming a single database, or select one with `--db PATH`.

## Multiple Databases

Use `lancedb.databases` to name local or remote databases that should be searched together:

```yaml
lancedb:
  databases:
    papers: s3://my-bucket/papers.lancedb
    wiki: s3://my-bucket/wiki.lancedb
    notes: /data/notes.lancedb
```

A location can be a URI or local path.

Results, documents, and citations use the configured name as `source`. An unavailable configured database raises `SourceUnavailableError`, which names the database and not its location, so a location never travels in an error a consumer might render or log. A migration, configuration or read-only failure keeps its own type, with the database named in the message. Commands that report on a database, such as `info`, still show where it is.

Searches spanning multiple databases identify each result with a model-facing `Collection:` line. Searches over one database omit it. Structured `source` fields on results, documents, citations, and analysis dictionaries are unchanged.

Embedding compatibility is checked against two different things.

On open, each database is compared with the current configuration. A dimension mismatch raises `ConfigMismatchError`. A provider or model-name mismatch at the same dimension warns in read-only mode and raises in writable mode.

Across a selection, the databases are compared with each other. Vector and hybrid search embed the query once, so every database answering it must record the same provider, model, and dimension. A disagreement raises `ConfigMismatchError` in read-only mode as well. Only the databases searched together have to agree, and full-text search embeds nothing, so it is unaffected.

### Search and Provenance

`search`, `ask`, and `analyze` use the full set by default. Pass `sources` to select a subset:

```python
results = await client.search("query")                     # every database
results = await client.search("query", sources=["papers"])   # one of them
```

Candidates are combined into one ranked list with the configured reranker, or by cosine similarity to the query when reranking is disabled, with within-database rank breaking ties (full-text-only searches order by retrieval score). `SearchResult.source`, `Citation.source`, and `Document.source` carry the database name, for a set and for one database alike.

The CLI labels results and citations only when the operation spans multiple databases. A command already narrowed with `--db-name` does not repeat the name on every result.

#### Duplicate IDs

IDs are unique within a database, not across databases. Copies of a database therefore retain the same IDs.

Citation ambiguity is evaluated against evidence available to the run. A cited chunk ID is rejected with `AmbiguousCitationError` if search returned it from multiple databases, or it was previously cited from another database. If only one retrieved result has the ID, that result is cited. For an ID absent from search results, the fallback checks every selected database and rejects multiple holders. A shared ID that nothing cites is ignored.

`get_document_by_id`, `get_chunk_by_id` and `get_picture_bytes` take an optional `source`, and ask that database alone. A name the client does not cover raises `UnknownDatabaseError`. Without one, the document and chunk lookups ask every covered database and answer from the first that holds the ID; `get_picture_bytes` requires one whenever the client covers a set.

The analysis sandbox rejects shared document IDs because its mount path is `/documents/{id}/`.

The chat document filter selects by document and database: the search is narrowed to the databases the selection names, and the ID filter applies within them. An ID that copies share still matches in every selected database that holds it.

#### Ranking

Without a reranker, the fused list is ordered by cosine similarity between the query vector and each candidate. The databases in a selection share an embedder, so similarity in that one space is comparable across databases, where retrieval scores are each database's own arithmetic. Ties resolve by the candidate's rank within its own database, and configured order decides only when both tie. Similarities rarely tie exactly, so declaration order decides almost nothing: on MTRAG retrieval benchmarks, reversing it left recall unchanged in every cell. Full-text-only searches have no query vector and order by retrieval score instead.

Results are not guaranteed to spread across databases: a database with nothing relevant to a query contributes nothing, and a strong database can fill every slot. On MTRAG retrieval benchmarks over two to eight collections, cosine fusion holds recall roughly flat as collections are added, where position-based fusion lost up to half its recall at eight.

A configured reranker scores the combined candidate set directly, ignoring which database each candidate came from, and remains the strongest option: roughly 6 to 8 recall points above cosine fusion on the same benchmarks. Its cost grows with the number of databases because each contributes candidates.

Image queries are vector-only and skip the reranker: the reranker interface takes a text query, and multimodal reranking applies to pictures on the candidate side, not to image queries. Their fused list is ordered by cosine similarity like any other vector search.

If a selected database is unavailable, the operation fails with `SourceUnavailableError`, which names that database.

### Python Operations

Creating, writing, rebuilding, and vacuuming require one database. Calling these operations on a client that covers multiple raises `AmbiguousDatabaseError`. Select one at creation time or obtain a single-database client:

```python
async with HaikuRAG(config=config, create=True, sources=["papers"]) as papers:
    ...

async with HaikuRAG(config=config) as client:
    papers = (await client.clients_for(["papers"]))[0]
```

Conversion, chunking, and title generation do not access a database and remain available on a multi-database client.

### CLI Commands

Commands use database sets as follows:

- **Set-capable**: `search`, `ask`, `analyze`, and `chat` use the full configured set, or the single database selected by `--db-name`.
- **Config-only**: `settings`, `init-config`, and `download-models` do not open a database.
- **Single-database**: everything else — document writes, `rebuild`, `vacuum`, `migrate`, `init`, `info`, `history`, `tag`, `doctor`, `list`, `inspect`, `visualize`, and `mcp` — works on one database, selected with the global `--db-name` option.

```bash
haiku-rag search "query"         # every configured database
haiku-rag --db-name papers list  # one of them
haiku-rag --db-name papers migrate
```

`--db-name` selects an entry from `lancedb.databases`, including remote entries, and `haiku.rag` when nothing is configured. `--db` opens a local path, named by its stem, whatever is configured. A single-database command requires one of these options when multiple databases are configured. A configured set of one is selected automatically.

Each database is created, migrated and vacuumed on its own:

```bash
haiku-rag --db-name papers init
haiku-rag --db-name wiki init
```

## Vector Indexing

Configure vector search settings:

```yaml
search:
  vector_index_metric: cosine  # cosine or l2
  vector_refine_factor: 30     # Re-ranking factor for accuracy
  vector_nprobes: 20           # IVF partitions searched per query
```

For search behavior settings (`limit`, `max_context_chars`), see [Search and Question Answering](qa.md#search-settings).

- **vector_index_metric**: Distance metric for vector similarity:
  - `cosine`: Cosine similarity (default, best for most embeddings)
  - `l2`: Euclidean distance
- **vector_refine_factor**: Improves accuracy when using a vector index by retrieving `refine_factor * limit` candidates (using approximate search) and re-ranking them with exact distances. Higher values increase accuracy but slow down queries. Default: 30
  - **Only applies with a vector index** - has no effect on brute-force search, which already returns exact results
- **vector_nprobes**: How many IVF partitions each query searches. Higher values increase recall and latency. A larger corpus holds more partitions, so the same value covers a smaller fraction of it. Default: 20
  - **Only applies with a vector index** - ignored by brute-force search

!!! note
    Vector indexes are only necessary for large datasets with over 100,000 chunks. For smaller datasets, LanceDB's brute-force kNN search provides exact results with good performance. Only create an index if you notice search performance degradation on large datasets.

Retrieval MAP with and without an index, measured on copies of the benchmark databases with no reranker:

| Dataset | Chunks | Dim | Exact | Indexed | Delta | Build | Peak RSS |
|---------|-------:|----:|------:|--------:|------:|------:|---------:|
| `hotpotqa` | 70,527 | 2560 | 0.6978 | 0.6979 | +0.0001 | 29.3 s | 3.19 GB |
| `orb_multimodal_nemotron` | 121,168 | 2048 | 0.9799 | 0.9800 | +0.0001 | 25.8 s | 3.38 GB |
| `frames` | 425,940 | 2560 | 0.5431 | 0.5387 | -0.0044 | 34.1 s | 4.02 GB |

An index costs no accuracy at 70k and 121k chunks and 0.0044 MAP at 426k. A larger corpus holds more IVF partitions, so the default number of probes covers a smaller fraction of the space, and `vector_refine_factor` can only re-score what those probes returned. Raise `vector_nprobes` to trade latency for recall on a large corpus. Build cost is near-flat in row count because training samples the data rather than scanning it, and vector dimension drives it more than corpus size.

**Index creation:**

Vector indexes are **not created automatically** during document ingestion to avoid slowing down the process. After you've added documents (at least 256 chunks required), create the index manually:

```bash
haiku-rag create-index
```

This command:
- Checks if you have enough data (minimum 256 chunks)
- Creates an IVF_PQ index for fast approximate nearest neighbor (ANN) search
- Uses LanceDB's automatic parameter calculation based on your dataset size and vector dimensions

**Re-indexing:**

New chunks reach the index without a rebuild. `optimize()`, which runs after writes while `auto_vacuum` is on, adds them as a delta part. Between a write and the next optimize, LanceDB serves ANN over the indexed rows and a brute-force scan over the remainder, then combines the results.

A rebuild retrains the centroids, which are fitted once at build time and never recomputed. As a corpus grows past the distribution it was trained on the partitioning fits it less well, and delta parts accumulate. Rebuild after substantial growth:

```bash
haiku-rag create-index
```

For datasets with fewer than 256 chunks, searches use brute-force kNN scans (exact nearest neighbors, 100% recall) which work well for small datasets but don't scale beyond a few hundred thousand vectors.
