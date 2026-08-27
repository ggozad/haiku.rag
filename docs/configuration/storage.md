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

This is an upstream limitation rather than a `haiku.rag` setting. Compaction bounds itself by row count instead of bytes, and LanceDB's async API exposes no batch size or fragment target to override it. Tracked at [lancedb/lancedb#2325](https://github.com/lancedb/lancedb/issues/2325). The requirement above will drop once compaction batches by bytes.

### Changing the Default Database Path

`storage.data_dir` holds the default database, always called
`haiku.rag.lancedb`. To put the database somewhere else for every command, give
`lancedb.uri` a local path:

```yaml
lancedb:
  uri: /data/notes.lancedb
```

An explicit `--db PATH` overrides `lancedb.uri` for that invocation.

This places one database without naming it. Its `source` is `None` in search
results, citations and documents, since only [`lancedb.databases`](#multiple-databases)
assigns the names that carry provenance. A path here changes where the database
lives, not what it is called.

A value with no scheme is a local path wherever it is configured, so
`haiku-rag init` creates it and every command that opens an existing database
requires it to exist. A mistyped path fails rather than becoming a new empty
database.

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

Operations on non-existent databases raise `FileNotFoundError`. This prevents accidental database creation from typos or misconfigured paths.

## Remote Storage

For remote storage, use the `lancedb` settings with various backends:

```yaml
# LanceDB Cloud
lancedb:
  uri: db://your-database-name
  api_key: your-api-key
  region: us-west-2  # optional

# Amazon S3
lancedb:
  uri: s3://my-bucket/my-table
  storage_options:
    region: us-east-1

# Amazon S3 with explicit credentials
lancedb:
  uri: s3://my-bucket/my-table
  storage_options:
    aws_access_key_id: YOUR_ACCESS_KEY
    aws_secret_access_key: YOUR_SECRET_KEY
    region: us-east-1

# S3-compatible (SeaweedFS, Tigris, etc.)
lancedb:
  uri: s3://my-bucket/my-table
  storage_options:
    endpoint: http://localhost:8333
    aws_access_key_id: YOUR_ACCESS_KEY
    aws_secret_access_key: YOUR_SECRET_KEY
    region: us-east-1
    allow_http: "true"

# Azure Blob Storage
lancedb:
  uri: az://my-container/my-table

# Google Cloud Storage
lancedb:
  uri: gs://my-bucket/my-table

# HDFS
lancedb:
  uri: hdfs://namenode:port/path/to/table
```

- **LanceDB Cloud** (`db://`): Requires `api_key` and `region`. Table optimization and indexing are managed server-side.
- **Object storage** (`s3://`, `gs://`, `az://`, `hdfs://`): Uses `storage_options` for credentials and endpoint configuration. Authentication can also be provided via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.) or cloud provider SDK defaults (AWS CLI, Azure CLI, gcloud).
- **S3-compatible stores** (MinIO, Tigris, etc.): Set `endpoint` in `storage_options`. When using `http://` endpoints, also set `allow_http: "true"`.
- **Local path** (no scheme): `uri` also takes a local path, which is how the default database is pointed elsewhere. See [Changing the Default Database Path](#changing-the-default-database-path).

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
- **Consumer processes** (1..N) — IAM role with `s3:Get/List` on the LanceDB bucket only. Run `haiku-rag --read-only mcp`, the chat TUI, etc. They never see the documents bucket.

Each process picks up its own credentials from the AWS default chain (env vars, IAM instance role, AWS profile), so no credentials are hard-coded in the configuration files.

## Multiple Databases

Use `lancedb.databases` to name local or remote databases that should be searched
together:

```yaml
lancedb:
  databases:
    papers: s3://my-bucket/papers.lancedb
    wiki: s3://my-bucket/wiki.lancedb
    notes: /data/notes.lancedb
```

A location can be a URI or local path. `databases` and `uri` are mutually
exclusive.

Results, documents, citations, and model context use the configured name as
`source`. Commands such as `info` and path-related errors still show locations.

Embedding compatibility is checked against two different things.

On open, each database is compared with the current configuration. A dimension
mismatch raises `ConfigMismatchError`. A provider or model-name mismatch at the
same dimension warns in read-only mode and raises in writable mode.

Across a selection, the databases are compared with each other. Vector and
hybrid search embed the query once, so every database answering it must record
the same provider, model, and dimension. A disagreement raises
`ConfigMismatchError` in read-only mode as well. Only the databases searched
together have to agree, and full-text search embeds nothing, so it is
unaffected.

### Search and Provenance

`search`, `ask`, and `analyze` use the full set by default. Pass `sources` to
select a subset:

```python
results = await client.search("query")                     # every database
results = await client.search("query", sources=["papers"])   # one of them
```

Candidates are combined into one ranked list with the configured reranker, or
with reciprocal rank fusion when reranking is disabled. `SearchResult.source`,
`Citation.source`, and `Document.source` contain the configured database name.
The name is retained when a client covers only one named database. Databases
configured through `lancedb.uri` are unnamed, so their `source` is `None`.

The CLI labels results and citations only when the operation spans multiple
databases. A command already narrowed with `--db-name` does not repeat the name
on every result.

#### Duplicate IDs

IDs are unique within a database, not across databases. Copies of a database
therefore retain the same IDs. Citation resolution raises
`AmbiguousCitationError` when a cited chunk ID exists in more than one selected
database; a shared ID that nothing cites is ignored. The analysis sandbox
rejects shared document IDs because its mount path is `/documents/{id}/`.

The chat document filter selects by document ID and applies `id IN (...)` to
every covered database, so selecting an ID that copies share matches the
document in each of them.

#### Ranking

Configure a reranker when searching multiple databases. Reciprocal rank fusion
compares positions rather than scores, so each database contributes top-ranked
results even when another database has stronger matches. A reranker scores the
combined candidate set directly.

In a 3,045-query evaluation over a corpus split across three databases,
reranking produced retrieval MAP 0.9914, compared with 0.9918 for the same corpus
in one database. Without a reranker, MAP was 0.6044, compared with 0.9798 in one
database. Reranking cost grows with the number of databases because each
contributes candidates.

Without a reranker, consider increasing `search.limit` with the number of
databases. With three complete rankings and a limit of 5, a database may
contribute only one or two results. A higher limit also sends more results to
the caller and model.

If any selected database fails to open, the operation fails and identifies that
database.

### Python Operations

Creating, writing, rebuilding, and vacuuming require one database. Calling these
operations on a client that covers multiple raises `AmbiguousDatabaseError`.
Select one at creation time or obtain a single-database client:

```python
async with HaikuRAG(config=config, create=True, sources=["papers"]) as papers:
    ...

async with HaikuRAG(config=config) as client:
    papers = (await client.clients_for(["papers"]))[0]
```

Conversion, chunking, and title generation do not access a database and remain
available on a multi-database client.

### CLI Commands

Commands use database sets as follows:

- **Set-capable**: `search`, `ask`, `analyze`, and `chat` use the full
  configured set, or the subset named by `--db-name`.
- **Config-only**: `settings`, `init-config`, and `download-models` do not open
  a database.
- **Single-database**: everything else — document writes, `rebuild`, `vacuum`,
  `migrate`, `init`, `info`, `history`, `tag`, `doctor`, `list`, `inspect`,
  `visualize`, and `mcp` — works on one database, selected with the global
  `--db-name` option.

```bash
haiku-rag search "query"         # every configured database
haiku-rag --db-name papers list  # one of them
haiku-rag --db-name papers migrate
```

`--db-name` selects an entry from `lancedb.databases`, including remote entries.
`--db` selects a local path and overrides the configured location. A
single-database command requires one of these options when multiple databases are
configured. A configured set of one is selected automatically.

Each database is created, migrated and vacuumed on its own:

```bash
haiku-rag --db-name papers init
haiku-rag --db-name wiki init
```

## Vector Indexing

Configure vector search settings:

```yaml
search:
  vector_index_metric: cosine  # cosine, l2, or dot
  vector_refine_factor: 30     # Re-ranking factor for accuracy
```

For search behavior settings (`limit`, `max_context_chars`), see [Search and Question Answering](qa.md#search-settings).

- **vector_index_metric**: Distance metric for vector similarity:
  - `cosine`: Cosine similarity (default, best for most embeddings)
  - `l2`: Euclidean distance
  - `dot`: Dot product similarity
- **vector_refine_factor**: Improves accuracy when using a vector index by retrieving `refine_factor * limit` candidates (using approximate search) and re-ranking them with exact distances. Higher values increase accuracy but slow down queries. Default: 30
  - **Only applies with a vector index** - has no effect on brute-force search, which already returns exact results

!!! note
    Vector indexes are only necessary for large datasets with over 100,000 chunks. For smaller datasets, LanceDB's brute-force kNN search provides exact results with good performance. Only create an index if you notice search performance degradation on large datasets.

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

Indexes are not automatically updated when you add new documents. After adding a significant amount of new data:

```bash
haiku-rag create-index  # Rebuilds the index with all data
```

Searches still work with stale indexes - LanceDB uses the index for old data (fast ANN) and brute-force kNN for new unindexed rows, then combines the results. However, performance degrades as more unindexed data accumulates.

For datasets with fewer than 256 chunks, searches use brute-force kNN scans (exact nearest neighbors, 100% recall) which work well for small datasets but don't scale beyond a few hundred thousand vectors.
