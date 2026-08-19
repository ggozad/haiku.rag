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
  compaction_target_bytes: 2147483648  # Target size for a compacted fragment
```

- **data_dir**: Directory for local database storage. When empty, uses platform-specific default locations
- **auto_vacuum**: When enabled (default), automatically runs vacuum after document create/update/delete operations and database rebuilds. Background vacuums are throttled to at most one every 5 minutes, so sustained ingestion does not trigger continuous compaction, and a final vacuum runs when the client closes. Set to `false` to disable automatic vacuuming and rely on manual `haiku-rag vacuum` commands only. Disabling can help avoid potential crashes in high-concurrency scenarios
- **vacuum_retention_seconds**: When vacuum runs, old table versions older than this threshold are removed. Default: 86400 seconds (1 day). Set to 0 for aggressive cleanup (removes all old versions immediately)
- **compaction_target_bytes**: Target size for the fragments compaction writes on the tables that store docling blobs. Default: 2 GiB. Advisory rather than a cap, see [Vacuum Memory](#vacuum-memory) below

!!! warning "Vacuum Retention Threshold"
    The `vacuum_retention_seconds` value should be larger than the typical time it takes to process and write a document. If a concurrent operation is in progress while vacuum runs, setting this value too low can cause race conditions where vacuum removes table versions that an in-flight operation still needs. The default of 86400 seconds (1 day) is conservative and safe for most use cases.

### Vacuum Memory

Vacuum compacts small data files into larger ones. LanceDB targets roughly one million rows per fragment, which a `documents` table holding multi-megabyte docling blobs never reaches, so an unsized pass re-merges the whole existing fragment rather than only the new rows, and peak memory scales with the table rather than with what was added.

The `documents` and `document_items` tables are therefore compacted with an explicit fragment target derived from `compaction_target_bytes`. The target is sized from the widest fragment's bytes per row, taken from table metadata without reading any payload, so a handful of very large documents shrinks it for the whole table. The remaining tables use LanceDB's own optimize, which is already bounded for them because their rows are small.

!!! warning "The target is not a memory cap"
    `compaction_target_bytes` sizes the fragments compaction **writes**. It cannot shrink a fragment that is already larger, and LanceDB rewrites such a fragment in one piece, costing roughly its own size no matter how low the target is set.

    Oversized fragments come from two places: a single large ingest batch, since each write becomes one fragment, and any database vacuumed before this release. In both cases the cost is paid once per fragment, the first time deletions within it pass LanceDB's 10% threshold, after which it is split to the target and stays there.

    So the practical peak is the larger of `compaction_target_bytes` and your biggest existing fragment. To lower the first, reduce it; to lower the second, ingest in smaller batches.

A row larger than the target cannot be sized at all: the target floors at one row and the pass costs roughly that row's size. A 300-page PDF at `images_scale: 2.0` produces a row of around 445 MB. To reduce row size rather than raise the target:

- Reduce `images_scale` (see [Image Settings](processing.md#image-settings)). Rendered page rasters dominate the size of `documents`, and their byte cost falls with the square of the scale factor.
- Set `generate_page_images: false` if visual grounding through `visualize_chunk()` is not needed. This removes page rasters entirely.

If fragment sizes are missing from the table metadata, which can happen for databases written by much older versions, compaction is skipped for that table and a warning is logged. Old versions are still pruned.

#### The first vacuum after upgrading

A database written before this release may contain one large fragment built by unsized compaction. It is left alone until deletions within it pass the 10% threshold, so early vacuums are cheap but its superseded payload is not yet reclaimed and disk usage can sit above the live data size. The pass that crosses the threshold rewrites it once, splitting it to the target, after which the table stays at the target and disk returns to normal.

Compaction options are not exposed by LanceDB's async API ([lancedb/lancedb#2325](https://github.com/lancedb/lancedb/issues/2325)), so these tables are compacted through `lance` directly.

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
