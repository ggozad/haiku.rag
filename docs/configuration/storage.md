# Database and Storage

## Local Storage

By default, `haiku.rag` uses a local LanceDB database:

```yaml
storage:
  data_dir: /path/to/data  # Empty = use default platform location
  auto_vacuum: true  # Enable automatic vacuuming after operations
  vacuum_retention_seconds: 86400  # Cleanup threshold in seconds
```

- **data_dir**: Directory for local database storage. When empty, uses platform-specific default locations
- **auto_vacuum**: When enabled (default), automatically runs vacuum after document create/update operations and database rebuilds. Set to `false` to disable automatic vacuuming and rely on manual `haiku-rag vacuum` commands only. Disabling can help avoid potential crashes in high-concurrency scenarios
- **vacuum_retention_seconds**: When vacuum runs, old table versions older than this threshold are removed. Default: 86400 seconds (1 day). Set to 0 for aggressive cleanup (removes all old versions immediately)

!!! warning "Vacuum Retention Threshold"
    The `vacuum_retention_seconds` value should be larger than the typical time it takes to process and write a document. If a concurrent operation is in progress while vacuum runs, setting this value too low can cause race conditions where vacuum removes table versions that an in-flight operation still needs. The default of 86400 seconds (1 day) is conservative and safe for most use cases.

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
# Use AWS credentials or IAM roles

# Azure Blob Storage
lancedb:
  uri: az://my-container/my-table
# Use Azure credentials

# Google Cloud Storage
lancedb:
  uri: gs://my-bucket/my-table
# Use GCP credentials

# HDFS
lancedb:
  uri: hdfs://namenode:port/path/to/table
```

Authentication is handled through standard cloud provider credentials (AWS CLI, Azure CLI, gcloud, etc.) or by setting `api_key` for LanceDB Cloud.

**Note:** Table optimization is automatically handled by LanceDB Cloud (`db://` URIs) and is disabled for better performance. For object storage backends (S3, Azure, GCS), optimization is still performed locally.

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

## Vector Indexing

Configure vector search settings:

```yaml
search:
  vector_index_metric: cosine  # cosine, l2, or dot
  vector_refine_factor: 30     # Re-ranking factor for accuracy
```

For search behavior settings (`limit`, `context_radius`, `max_context_items`, `max_context_chars`), see [QA and Research](qa-research.md#search-settings).

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
