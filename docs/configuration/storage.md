# Database and storage

## Operational constraints

Five things to know before deploying.

**Run one writer per database.** This is a haiku.rag constraint, not a LanceDB
one. A write that spans several tables is serialized by an in-process lock and
rolled back by restoring each table to the version it had when the write started.
Both are process-local: a second writing process can commit between that snapshot
and the mutation, and a rollback would then revert its work too. Run
a single writer, either the [`haiku-ingester`](../ingester/index.md) service or your own
application. Read-only consumers are unrestricted.

**Readers lag by an interval.** A connection always sees its own writes. It sees
another process's writes after `lancedb.read_consistency_interval_seconds`
(default 30).

**Migrate after an upgrade that changes the schema.** `haiku-rag migrate` applies
pending migrations in place, and `haiku-rag info` lists what is pending. A
release that needs it says so in the [changelog](../changelog.md). A release
refuses to write to a database whose recorded `schema_version` is newer than
its own schema. A database recorded before 0.64.0 is migrated with haiku.rag
0.111.1 first (see [`migrate`](../cli.md#migrate)).

**The embedding dimension is fixed per database.** Every chunk vector has the
dimension the database was created with. Changing `embeddings.model.vector_dim`
raises `ConfigMismatchError` on open, because stored vectors cannot be compared
against new ones. Changing the provider, the model name or a prefix while keeping
the dimension warns on a read-only open and raises on a writable one, and so does a full
rebuild, `--rechunk` or `--descriptions` to another embedder at the same
dimension that was interrupted. `haiku-rag rebuild --embed-only` re-embeds with
the configured model, and running an interrupted rebuild again finishes it.
`haiku-rag rebuild --set-embedder` records the configured identity without
re-embedding: use it only for an embedder that produces the same vectors as the
recorded one, such as the same model served by another stack. It records a
changed query prefix and refuses a changed document prefix.

**The database records only its version and embedder.** The `settings` table
holds the haiku.rag version that last wrote or migrated the database, and the
embedder's `provider`, `name` and `vector_dim`, plus `query_prefix` and
`document_prefix` when set. Every open checks these against the configuration.
Nothing else from the configuration is stored, so a database can be copied or shared without the credentials of the process that wrote it.
Databases written before 0.89.0 stored the whole configuration. `haiku-rag
migrate` reduces it, and the older table versions still hold it. To remove them,
stop every process using the database, run `haiku-rag migrate`, delete the tags
taken before it, and run `haiku-rag vacuum --retention-seconds 0`.
The default retention keeps them for a day, and a tag keeps its version.

## Local storage

By default, `haiku.rag` uses a local LanceDB database:

```yaml
storage:
  data_dir: /path/to/data  # Empty = use default platform location
  auto_vacuum: true  # Enable automatic vacuuming after operations
  vacuum_retention_seconds: 86400  # Cleanup threshold in seconds
  compaction_target_bytes: 2147483648  # Target size for a compacted fragment
  store_originals: true  # Store PDF and image files for visual grounding
```

- **data_dir**: Directory for local database storage. When empty, uses platform-specific default locations
- **auto_vacuum**: When enabled (default), automatically runs vacuum after document create/update/delete operations and database rebuilds. Background vacuums are throttled to at most one every 5 minutes, so sustained ingestion does not trigger continuous compaction, and a final vacuum runs when the client closes. Set to `false` to vacuum only with `haiku-rag vacuum`. Vacuum is also what adds newly written rows to the indexes, so with `false` they stay outside them until `haiku-rag vacuum`, or `haiku-rag rebuild --reindex` for the full-text and scalar indexes. With the ingester's [maintenance](../ingester/running.md#maintenance) enabled, the ingester's clients ignore it and the maintenance pass vacuums
- **vacuum_retention_seconds**: When vacuum runs, old table versions older than this threshold are removed. Default: 86400 seconds (1 day). Set to 0 for aggressive cleanup (removes all old versions immediately)
- **compaction_target_bytes**: Target size for the fragments compaction writes on the tables that store docling blobs. Default: 2 GiB. Advisory rather than a cap, see [Vacuum memory](#vacuum-memory) below
- **store_originals**: When enabled (default), ingesting a PDF or an image stores the file in the `documents` table, and visual grounding renders its pages from it. When disabled, no file is stored and documents ingested meanwhile have no visual grounding until `haiku-rag rebuild --originals` stores their files. The setting governs writes only: a stored file renders either way

!!! warning "Vacuum retention"
    Keep `vacuum_retention_seconds` above the time a document takes to process and write. A lower value lets vacuum remove table versions an in-flight operation still needs.

### Vacuum memory

Vacuum compacts small data files into larger ones. The `documents` and `document_items` tables, which hold the docling blobs, are compacted to fragments of about `compaction_target_bytes`, sized from the widest fragment's bytes per row. The other tables are compacted with LanceDB's default fragment size.

!!! warning "The target is not a memory cap"
    `compaction_target_bytes` sizes the fragments that compaction **writes**. It cannot shrink a fragment that is already larger. LanceDB rewrites such a fragment in one piece, at roughly its own size, the first time deletions within it pass 10%, and splits it to the target. A single large ingest batch writes one large fragment.

    Peak memory is the larger of `compaction_target_bytes` and the biggest existing fragment. Lower the first by reducing it, and the second by ingesting in smaller batches.

A row larger than the target costs roughly its own size. A `documents` row holds the text, the compressed docling structure without images, and the original PDF or image file, so it is about the size of the file. `store_originals: false` leaves the file out.

Vacuum also adds new rows to the full-text index. Search stays correct without it but scans the uncovered rows on every query. `haiku-rag doctor` reports the coverage.

When a table's metadata has no fragment sizes, compaction skips that table with a warning. Old versions are still pruned.

### Placing the database

`lancedb.databases` maps a name to a location, a local path or a URI, or to a location read at a tag (see [Reading at a tag](multiple-databases.md#reading-at-a-tag)), and is the one way to place databases. With nothing configured, the database is the entry `haiku.rag` at `<storage.data_dir>/haiku.rag.lancedb`. To put one database somewhere else, name it:

```yaml
lancedb:
  databases:
    notes: /data/notes.lancedb
```

The name is what `source` carries in search results, citations and documents, and what `--db-name` and `sources` select. The default database answers to `haiku.rag`.

An explicit `--db PATH` on the command line opens that database instead, named by the path's stem, whatever is configured. From Python, `db_path` places the database only where the configuration places none: beside `lancedb.databases` it raises `AmbiguousDatabaseError`.

A value with no scheme is a local path wherever it is configured, so `haiku-rag init` creates it and every command that opens an existing database requires it to exist. A mistyped path fails rather than becoming a new empty database.

## Database creation

Databases must be explicitly created before use:

**CLI:**
```bash
# Create in the default location
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

Opening a nonexistent local database given as a path raises `FileNotFoundError`, naming the path. A configured or default database raises `SourceUnavailableError` instead, naming the database and not its location.

## Remote storage

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

- **LanceDB Cloud** (`db://`): Requires `api_key` and `region`. LanceDB Cloud optimizes and indexes tables itself, so haiku.rag does not.
- **Object storage** (`s3://`, `gs://`, `az://`, `hdfs://`): Uses `storage_options` for credentials and endpoint configuration. Authentication can also be provided via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.) or cloud provider SDK defaults (AWS CLI, Azure CLI, gcloud).
- **S3-compatible stores** (MinIO, Tigris, etc.): Set `endpoint` in `storage_options`. When using `http://` endpoints, also set `allow_http: "true"`.
- **Local path** (no scheme): see [Placing the database](#placing-the-database).

The `storage_options` keys are case-insensitive and passed directly to the underlying object store library. Available keys depend on the backend. See the [LanceDB storage docs](https://lancedb.com/docs/storage/) for details.

### Caching and read consistency

```yaml
lancedb:
  read_consistency_interval_seconds: 30   # null to never re-check
  index_cache_size_bytes: 536870912       # default null: the LanceDB default
  metadata_cache_size_bytes: 268435456    # default null
```

- **read_consistency_interval_seconds**: how often a connection checks for writes from another process. `null` never checks, so a long-lived reader never sees the ingester's writes. `0` checks on every read.
- **index_cache_size_bytes** / **metadata_cache_size_bytes**: sizes for the caches held by the LanceDB session, which is shared across every connection in the process. The first vector query loads the index into it, so on object storage the cache is what stops the next connection refetching it. Size it for the total set of indexes a process keeps warm, against the memory available to it.

### Deployment pattern: one writer, many readers

The [one-writer constraint](#operational-constraints) shapes the deployment: one
writing process per database URI, any number of read-only consumers.

The recommended layout for production is "different buckets, same account, separate IAM roles per process":

- **Ingestion process**: IAM role with `s3:Get/List` on the documents bucket and `s3:Get/Put/Delete` on the LanceDB bucket. Runs `haiku-ingester serve` (with an `s3` source in `ingester.databases` pointing at the documents bucket). Exactly one such process per LanceDB URI.
- **Consumer processes** (1..N): IAM role with `s3:Get/List` on the LanceDB bucket only. Run `haiku-rag mcp`, the chat TUI, etc. They never see the documents bucket.

Each process picks up its own credentials from the AWS default chain (env vars, IAM instance role, AWS profile), so no credentials are hard-coded in the configuration files.

`haiku-ingester` writes every database `ingester.databases` names, see [Multiple databases](multiple-databases.md#ingester).

## Multiple databases

Several databases can be configured and searched together. See [Multiple databases](multiple-databases.md).

## Vector indexing

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
  - Only applies with a vector index. Brute-force search already returns exact results
- **vector_nprobes**: How many IVF partitions each query searches. Higher values increase recall and latency. A larger corpus holds more partitions, so the same value covers a smaller fraction of it. Default: 20
  - Only applies with a vector index

!!! note
    Below about 100,000 chunks, brute-force kNN search is exact and fast enough. Create an index when search slows on a larger corpus.

An index can cost recall on a large corpus. A larger corpus holds more IVF partitions, so the default probes cover less of it, and `vector_refine_factor` only re-scores what the probes returned. Raise `vector_nprobes` to trade latency for recall on a large corpus.

Writing documents never creates a vector index. The ingester's [maintenance](../ingester/running.md#maintenance) builds one when it is enabled. Without it, once the database holds at least 256 chunks, build one:

```bash
haiku-rag create-index
```

It builds an IVF_PQ index, with LanceDB choosing the parameters from the row count and vector dimension.

New chunks reach the index without a rebuild. Vacuum, which runs after writes while `auto_vacuum` is on, adds them as a delta part, and merges the parts into one once there are more than four. Every concurrent query opens files for each part. Between a write and the next vacuum, LanceDB serves ANN over the indexed rows and a brute-force scan over the remainder, then combines the results.

Vacuum also splits the index's partitions as the table grows, so a grown index needs no retraining and matches a fresh build's recall. A grown index keeps more than one delta part, so its queries are slower. `haiku-rag create-index` builds it again as one part. A full `rebuild`, `--rechunk`, `--embed-only` and `--descriptions` retrain an existing index, and no `rebuild` creates one.
