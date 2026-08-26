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
results, citations and documents, since only [`lancedb.databases`](#several-databases)
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

## Several Databases

`lancedb.databases` maps a name to a location, for searching several databases at
once:

```yaml
lancedb:
  databases:
    medic: s3://my-bucket/medic.lancedb
    st: s3://my-bucket/st.lancedb
    local: /data/notes.lancedb
```

A location is a URI or a local path. `databases` and `uri` are mutually
exclusive, and setting both fails validation.

Results, citations, model input and errors opening a named database carry the
configured name rather than the location, so a path or a bucket does not reach a
trace or a model. Commands that report on a database — `info`, `init`, `tag` —
print its location, as does an error about a path.

Every database in the set is opened with the same embedding configuration. A
different `vector_dim` raises `ConfigMismatchError` on open. A different provider
or model name at the same dimension is a warning on a read-only open, since the
same model served by another stack is spelled differently, and raises on a
writable one.

Searching embeds the query once for the whole selection, so the databases
searched together must have been written with the same embedder. Two that
disagree with each other raise `ConfigMismatchError` naming both, whatever the
configuration says.

### Searching a set

`search`, `ask` and `analyze` cover the whole set, or the subset named by
`sources`:

```python
results = await client.search("query")                     # every database
results = await client.search("query", sources=["medic"])   # one of them
```

Candidates from each database are fused into one ranked list, by the configured
reranker where there is one and by reciprocal rank fusion otherwise. Each result
carries `source`, the name of the database it came from, and so does each
citation. A document from `list_documents`, `get_document_by_id`,
`get_document_by_uri` or `resolve_document` carries it too, and a database
named in `lancedb.databases` keeps that name even when it is the only one a
client covers. Only a database placed by `lancedb.uri`, which names none, has no
name to carry. The CLI does not print the name of a single database it was told
to use, since the caller just named it.

Chunk ids are unique within a database and say nothing across them, so a database
copied from another holds the same ids. Results are told apart by the database
and the id together. A chunk id held by two of the databases searched cannot be
cited: `resolve_citations` raises `AmbiguousCitationError`, and the capability
asks the model for other evidence instead.

Document ids behave the same way, and two places treat a collision differently.
The analysis sandbox mounts one document per id and refuses a set where two
databases claim one, since the mount has a single path per id. The chat document
filter does not: it selects by document id and applies `id IN (...)` to every
covered database, so selecting an id that exists in copies of a database matches
the document in each of them. Independently built databases use UUID document
ids and do not collide.

**Configure a reranker when searching several databases.** Reciprocal rank fusion
compares ranks, not scores, so every database contributes its own best matches
whether or not they are relevant to the question, and results from databases
holding nothing relevant displace better ones. A reranker scores the whole union
instead, which removes the effect. Measured on one corpus split three ways, with
the same queries: retrieval MAP 0.9914 with a reranker against 0.9918 for the
same corpus in a single database, and 0.6044 without one against 0.9798. The cost
is that a reranker scores candidates in proportion to the number of databases.

Without one, consider raising `search.limit` with the number of databases
searched. Each contributes its own best matches to a list that is then truncated
at the limit, so with three full rankings and a limit of 5 any one database may
contribute only one or two results. Raising the limit raises what the caller and
the model receive, since without a reranker nothing is over-fetched to absorb
it.

Creating names a database: `create=True` on a client covering the set raises
`AmbiguousDatabaseError`, and `HaikuRAG(config=config, create=True,
sources=["name"])` creates that one.

Converting, chunking and title generation are functions of the configuration
rather than of a database, so they work on a client covering the set. Writing,
rebuilding and vacuuming name one database: asking a set-covering client raises
`AmbiguousDatabaseError`, and `client.clients_for(["name"])` returns a client for
one of them, writable when the covering client is.

A database that cannot be opened fails the whole query and is named in the error.
A result set silently missing one of the databases asked for cannot be told apart
from a complete one.

### How commands treat the set

Commands fall into three groups:

- **Set-capable**: `search`, `ask`, `analyze` and `chat` cover the whole
  configured set, or the subset named by `--db-name`.
- **Config-only**: `settings`, `init-config` and `download-models` open no
  database, so the set is irrelevant to them.
- **Single-database**: everything else — document writes, `rebuild`, `vacuum`,
  `migrate`, `init`, `info`, `history`, `tag`, `doctor`, `list`, `inspect`,
  `visualize` and `mcp` — works on one database, named with the global
  `--db-name` option.

```bash
haiku-rag search "query"                  # every configured database
haiku-rag --db-name medic list           # one of them
haiku-rag --db-name medic migrate
```

`--db-name` takes a name from `lancedb.databases`, which is how a database
behind a URI is reached. `--db` takes a path, and overrides the configured
location with that one database. A single-database command given neither fails
rather than choosing for you, unless `lancedb.databases` names exactly one: a
set of one is unambiguous and is used, keeping its configured name.

Each database is created, migrated and vacuumed on its own:

```bash
haiku-rag --db-name medic init
haiku-rag --db-name st init
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
