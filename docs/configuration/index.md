# Configuration

Configuration is done through YAML configuration files.

!!! note
    If you create a db with certain settings and later change them, `haiku.rag` will detect incompatibilities (for example, if you change embedding provider) and will exit. You can **rebuild** the database to apply the new settings, see [Rebuild Database](../cli.md#rebuild-database).

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
    enable_thinking: false
```

## Complete Configuration Example

```yaml
# haiku.rag.yaml
environment: production

storage:
  data_dir: ""  # Empty = use default platform location
  vacuum_retention_seconds: 86400

monitor:
  directories:
    - /path/to/documents
    - /another/path
  ignore_patterns: []  # Gitignore-style patterns to exclude
  include_patterns: []  # Gitignore-style patterns to include

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
    enable_thinking: false
  max_sub_questions: 3
  max_iterations: 2
  max_concurrency: 1

research:
  model:
    provider: ""  # Empty to use qa settings
    name: ""
    enable_thinking: false
  max_iterations: 3
  confidence_threshold: 0.8
  max_concurrency: 1

search:
  limit: 5                     # Default number of results to return
  context_radius: 0            # DocItems before/after to include for text content
  max_context_items: 10        # Maximum items in expanded context
  max_context_chars: 10000     # Maximum characters in expanded context
  vector_index_metric: cosine  # cosine, l2, or dot
  vector_refine_factor: 30

agui:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  cors_credentials: true
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["*"]

processing:
  converter: docling-local  # docling-local or docling-serve
  chunker: docling-local    # docling-local or docling-serve
  chunker_type: hybrid      # hybrid or hierarchical
  chunk_size: 256
  chunking_tokenizer: "Qwen/Qwen3-Embedding-0.6B"
  chunking_merge_peers: true
  chunking_use_markdown_tables: false
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

  vllm:
    embeddings_base_url: ""
    rerank_base_url: ""
    qa_base_url: ""
    research_base_url: ""

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
            temperature=0.7
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
client = HaikuRAG(config=custom_config)
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
- **[Search and Question Answering](qa-research.md)** - Search settings, question answering, and research workflows
- **[Document Processing](processing.md)** - Document conversion, chunking, and file monitoring
- **[Storage](storage.md)** - Database, remote storage, and vector indexing
