# Configuration

Configuration is done through YAML configuration files.

!!! note
    haiku.rag enforces one hard rule on existing databases: the embedding `vector_dim` in your config must match the value stored in the db. A mismatch exits with `ConfigMismatchError` and you must **rebuild** to apply the change (see [Rebuild Database](../cli.md#rebuild-database)).

    Opening a database never writes to it, so the stored embedding identity is left untouched. Changing only `provider` or `name` (e.g. switching from Ollama to vLLM serving the same model) is treated as soft drift: read-only opens log a warning and continue, while writable opens exit with `ConfigMismatchError`. Reconcile the stored identity with your config by running `haiku-rag rebuild --set-embedder` (see [Rebuild Database](../cli.md#rebuild-database)). If the change was unintentional, revert your config instead.

## Getting Started

Generate a configuration file with defaults:

```bash
haiku-rag init-config
```

This creates a `haiku.rag.yaml` file in your current directory with all available settings.

## Configuration File Locations

`haiku.rag` searches for configuration files in this order:

1. Path specified via `--config` flag: `haiku-rag --config /path/to/config.yaml <command>`
2. `./haiku.rag.yaml` (current directory)
3. Platform-specific user directory:
    - **Linux**: `~/.local/share/haiku.rag/haiku.rag.yaml`
    - **macOS**: `~/Library/Application Support/haiku.rag/haiku.rag.yaml`
    - **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/haiku.rag.yaml`

## Environment Variables

Any string value can reference an environment variable, so secrets stay out of the file and one config can serve multiple deployments:

```yaml
ingester:
  queue:
    dburi: postgresql+asyncpg://haiku:${POSTGRES_PASSWORD}@db:5432/haiku_rag
```

- `${VAR}` is replaced with the value of `VAR`. If `VAR` is unset, loading fails with an error naming the variable.
- `${VAR:-default}` uses `default` when `VAR` is unset or empty.
- `$$` produces a literal `$`.

Substitution happens after the YAML is parsed, so a value containing `:`, `@`, or `#` fills the string verbatim and never changes the document structure.

## Minimal Configuration

A minimal configuration file with defaults:

```yaml
# haiku.rag.yaml
environment: production

embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560

qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: true
```

## Complete Configuration Example

```yaml
# haiku.rag.yaml
environment: production

storage:
  data_dir: ""  # Empty = use default platform location
  vacuum_retention_seconds: 86400

ingester:
  sources:
    - type: fs
      id: local-docs
      root: /path/to/documents
      ignore_patterns: []  # Gitignore-style patterns to exclude
      include_patterns: []  # Gitignore-style patterns to include
      delete_orphans: true

lancedb:
  uri: ""  # Empty for local, or db://, s3://, az://, gs://
  api_key: ""
  region: ""

embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560

reranking:
  model:
    provider: ""  # Empty to disable, or mxbai, cohere, zeroentropy, vllm
    name: ""

qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: true
    temperature: 0.3
  max_searches: 3

search:
  limit: 10                    # Default number of results to return
  max_context_chars: 10000     # Maximum characters in expanded context
  vector_index_metric: cosine  # cosine, l2, or dot
  vector_refine_factor: 30

doctor:
  duplicates:                    # Near-duplicate document detection (doctor command)
    similarity_threshold: 0.97   # cosine cutoff on document embedding centroids
    min_chunks: 3                # documents with fewer chunks are excluded

prompts:
  domain_preamble: ""  # Prepended to skill instructions

processing:
  converter: docling-local  # docling-local or docling-serve
  chunker: docling-local    # docling-local or docling-serve
  chunker_type: hybrid      # hybrid or hierarchical
  chunk_size: 256
  chunking_tokenizer: "Qwen/Qwen3-Embedding-0.6B"
  chunking_merge_peers: true
  chunking_use_markdown_tables: false
  auto_title: false              # Auto-generate titles on ingestion
  title_model:
    provider: ollama
    name: gpt-oss
    enable_thinking: false
    temperature: 0.3
    max_tokens: 100
  conversion_options:
    do_ocr: true
    force_ocr: false
    ocr_lang: []
    do_table_structure: true
    table_mode: accurate
    table_cell_matching: true
    images_scale: 2.0

providers:
  ollama:
    base_url: http://localhost:11434

  docling_serve:
    base_url: http://localhost:5001
    api_key: ""
    timeout: 300
```

## Programmatic Configuration

When using haiku.rag as a Python library, you can pass configuration directly to the `HaikuRAG` client:

```python
from haiku.rag.config import AppConfig
from haiku.rag.config.models import EmbeddingModelConfig, ModelConfig, QAConfig, EmbeddingsConfig
from haiku.rag.client import HaikuRAG

# Create custom configuration
custom_config = AppConfig(
    qa=QAConfig(
        model=ModelConfig(
            provider="openai",
            name="gpt-4o",
            temperature=0.3
        )
    ),
    embeddings=EmbeddingsConfig(
        model=EmbeddingModelConfig(
            provider="ollama",
            name="qwen3-embedding:4b",
            vector_dim=2560
        )
    ),
    processing={"chunk_size": 512}
)

# Pass configuration to the client
async with HaikuRAG(config=custom_config) as client:
    ...
```

If you don't pass a config, the client uses the global configuration loaded from your YAML file or defaults.

This is useful for:
- Jupyter notebooks
- Python scripts
- Testing with different configurations
- Applications that need multiple clients with different configurations

## Configuration Topics

For detailed configuration of specific topics, see:

- **[Providers](providers.md)** - Model settings and provider-specific configuration (embeddings, reranking)
- **[Search and Question Answering](qa.md)** - Search settings and question answering
- **[Document Processing](processing.md)** - Document conversion and chunking
- **[Ingester](../ingester.md)** - Continuous ingestion from filesystem, HTTP, S3, and WebDAV sources
- **[Storage](storage.md)** - Database, remote storage, and vector indexing
- **[Prompts](prompts.md)** - Customize agent prompts for your domain
