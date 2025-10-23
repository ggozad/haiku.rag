# Configuration

Configuration is done through YAML configuration files.

!!! note
    If you create a db with certain settings and later change them, `haiku.rag` will detect incompatibilities (for example, if you change embedding provider) and will exit. You can **rebuild** the database to apply the new settings, see [Rebuild Database](./cli.md#rebuild-database).

## Getting Started

Generate a configuration file with defaults:

```bash
haiku-rag init-config
```

This creates a `haiku.rag.yaml` file in your current directory with all available settings.

To migrate from environment variables (`.env` file):

```bash
haiku-rag init-config --from-env
```

## Configuration File Locations

`haiku.rag` searches for configuration files in this order:

1. Path specified via `--config` flag: `haiku-rag --config /path/to/config.yaml <command>`
2. `./haiku.rag.yaml` (current directory)
3. `~/.config/haiku.rag/config.yaml` (user config directory)

## Minimal Configuration

A minimal configuration file with defaults:

```yaml
# haiku.rag.yaml
environment: production

embeddings:
  provider: ollama
  model: qwen3-embedding
  vector_dim: 4096

qa:
  provider: ollama
  model: gpt-oss
```

## Complete Configuration Example

```yaml
# haiku.rag.yaml
environment: production

storage:
  data_dir: ""  # Empty = use default platform location
  monitor_directories:
    - /path/to/documents
    - /another/path
  disable_autocreate: false
  vacuum_retention_seconds: 60

lancedb:
  uri: ""  # Empty for local, or db://, s3://, az://, gs://
  api_key: ""
  region: ""

embeddings:
  provider: ollama
  model: qwen3-embedding
  vector_dim: 4096

reranking:
  provider: ""  # Empty to disable, or mxbai, cohere, vllm
  model: ""

qa:
  provider: ollama
  model: gpt-oss

research:
  provider: ""  # Empty to use qa settings
  model: ""

processing:
  chunk_size: 256
  context_chunk_radius: 0
  markdown_preprocessor: ""

providers:
  ollama:
    base_url: http://localhost:11434

  vllm:
    embeddings_base_url: ""
    rerank_base_url: ""
    qa_base_url: ""
    research_base_url: ""

a2a:
  max_contexts: 1000
```

## API Keys

API keys are configured through **environment variables**, not in the YAML file.

```bash
# OpenAI
export OPENAI_API_KEY=your-key-here

# Anthropic
export ANTHROPIC_API_KEY=your-key-here

# Voyage AI
export VOYAGE_API_KEY=your-key-here

# Cohere
export CO_API_KEY=your-key-here
```

## File Monitoring

Set directories to monitor for automatic indexing:

```yaml
storage:
  monitor_directories:
    - /path/to/documents
    - /another_path/to/documents
```

## Embedding Providers

If you use Ollama, you can use any pulled model that supports embeddings.

### Ollama (Default)

```yaml
embeddings:
  provider: ollama
  model: mxbai-embed-large
  vector_dim: 1024
```

### VoyageAI

If you want to use VoyageAI embeddings you will need to install `haiku.rag` with the VoyageAI extras:

```bash
uv pip install haiku.rag[voyageai]
```

```yaml
embeddings:
  provider: voyageai
  model: voyage-3.5
  vector_dim: 1024
```

Set your API key via environment variable:

```bash
export VOYAGE_API_KEY=your-api-key
```

### OpenAI

OpenAI embeddings are included in the default installation:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small  # or text-embedding-3-large
  vector_dim: 1536
```

Set your API key via environment variable:

```bash
export OPENAI_API_KEY=your-api-key
```

### vLLM

For high-performance local inference, you can use vLLM to serve embedding models with OpenAI-compatible APIs:

```yaml
embeddings:
  provider: vllm
  model: mixedbread-ai/mxbai-embed-large-v1
  vector_dim: 512

providers:
  vllm:
    embeddings_base_url: http://localhost:8000
```

**Note:** You need to run a vLLM server separately with an embedding model loaded.

## Question Answering Providers

Configure which LLM provider to use for question answering. Any provider and model supported by [Pydantic AI](https://ai.pydantic.dev/models/) can be used.

### Ollama (Default)

```yaml
qa:
  provider: ollama
  model: gpt-oss

providers:
  ollama:
    base_url: http://localhost:11434
```

### OpenAI

OpenAI QA is included in the default installation:

```yaml
qa:
  provider: openai
  model: gpt-4o-mini  # or gpt-4, gpt-3.5-turbo, etc.
```

Set your API key via environment variable:

```bash
export OPENAI_API_KEY=your-api-key
```

### Anthropic

Anthropic QA is included in the default installation:

```yaml
qa:
  provider: anthropic
  model: claude-3-5-haiku-20241022  # or claude-3-5-sonnet-20241022, etc.
```

Set your API key via environment variable:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

### vLLM

For high-performance local inference:

```yaml
qa:
  provider: vllm
  model: Qwen/Qwen3-4B  # Any model with tool support in vLLM

providers:
  vllm:
    qa_base_url: http://localhost:8002
```

**Note:** You need to run a vLLM server separately with a model that supports tool calling loaded. Consult the specific model's documentation for proper vLLM serving configuration.

### Other Providers

Any provider supported by Pydantic AI can be used. Examples:

```yaml
# Google Gemini
qa:
  provider: gemini
  model: gemini-1.5-flash

# Groq
qa:
  provider: groq
  model: llama-3.3-70b-versatile

# Mistral
qa:
  provider: mistral
  model: mistral-small-latest
```

See the [Pydantic AI documentation](https://ai.pydantic.dev/models/) for the complete list of supported providers and models.

## Reranking

Reranking improves search quality by re-ordering the initial search results using specialized models. When enabled, the system retrieves more candidates (3x the requested limit) and then reranks them to return the most relevant results.

Reranking is **disabled by default** (`provider: ""`) for faster searches. You can enable it by configuring one of the providers below.

### MixedBread AI

For MxBAI reranking, install with mxbai extras:

```bash
uv pip install haiku.rag[mxbai]
```

Then configure:

```yaml
reranking:
  provider: mxbai
  model: mixedbread-ai/mxbai-rerank-base-v2
```

### Cohere

Cohere reranking is included in the default installation:

```yaml
reranking:
  provider: cohere
  model: rerank-v3.5
```

Set your API key via environment variable:

```bash
export CO_API_KEY=your-api-key
```

### vLLM

For high-performance local reranking using dedicated reranking models:

```yaml
reranking:
  provider: vllm
  model: mixedbread-ai/mxbai-rerank-base-v2

providers:
  vllm:
    rerank_base_url: http://localhost:8001
```

**Note:** vLLM reranking uses the `/rerank` API endpoint. You need to run a vLLM server separately with a reranking model loaded. Consult the specific model's documentation for proper vLLM serving configuration.

## Other Settings

### Database and Storage

By default, `haiku.rag` uses a local LanceDB database:

```yaml
storage:
  data_dir: /path/to/data  # Empty = use default platform location
```

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

#### Disable database auto-creation

By default, haiku.rag creates the local LanceDB directory and required tables on first use. To prevent accidental database creation and fail fast if a database hasn't been set up yet:

```yaml
storage:
  disable_autocreate: true
```

When enabled, for local paths, haiku.rag errors if the LanceDB directory does not exist, and it will not create parent directories.

### Document Processing

```yaml
processing:
  # Chunk size for document processing
  chunk_size: 256

  # Number of adjacent chunks to include before/after retrieved chunks for context
  # 0 = no expansion (default), 1 = include 1 chunk before and after, etc.
  # When expanded chunks overlap or are adjacent, they are automatically merged
  # into single chunks with continuous content to eliminate duplication
  context_chunk_radius: 0

  # Optional dotted path or file path to a callable that preprocesses
  # markdown content before chunking
  markdown_preprocessor: ""

storage:
  # Vacuum retention threshold (seconds) for automatic cleanup
  # When documents are added/updated, old table versions older than this are removed
  # Default: 60 seconds (safe for concurrent connections)
  # Set to 0 for aggressive cleanup (removes all old versions immediately)
  vacuum_retention_seconds: 60
```

#### Markdown Preprocessor

Optionally preprocess Markdown before chunking by pointing to a callable that receives and returns Markdown text. This is useful for normalizing content, stripping boilerplate, or applying custom transformations before chunk boundaries are computed.

```yaml
processing:
  # A callable path in one of these formats:
  # - package.module:func
  # - package.module.func
  # - /abs/or/relative/path/to/file.py:func
  markdown_preprocessor: my_pkg.preprocess:clean_md
```

!!! note
    - The function signature should be `def clean_md(text: str) -> str` or `async def clean_md(text: str) -> str`.
    - If the function raises or returns a non-string, haiku.rag logs a warning and proceeds without preprocessing.
    - The preprocessor affects only the chunking pipeline. The stored document content remains unchanged.

Example implementation:

```python
# my_pkg/preprocess.py
def clean_md(text: str) -> str:
    # strip HTML comments and collapse multiple blank lines
    lines = [line for line in text.splitlines() if not line.strip().startswith("<!--")]
    out = []
    for line in lines:
        if line.strip() == "" and (out and out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out)
```

## Migration from Environment Variables

!!! warning "Deprecation Notice"
    Environment variable configuration via `.env` files is deprecated and will be removed in future versions. Please migrate to YAML configuration.

To migrate your existing `.env` file to YAML:

```bash
haiku-rag init-config --from-env
```

This will read your current environment variables and generate a `haiku.rag.yaml` file with those settings.
