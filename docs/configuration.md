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
  provider: ollama
  model: qwen3-embedding
  vector_dim: 4096

reranking:
  provider: ""  # Empty to disable, or mxbai, cohere, zeroentropy, vllm
  model: ""

qa:
  provider: ollama
  model: gpt-oss

research:
  provider: ""  # Empty to use qa settings
  model: ""
  max_iterations: 3
  confidence_threshold: 0.8
  max_concurrency: 1

agui:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  cors_credentials: true
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["*"]

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
```

## Programmatic Configuration

When using haiku.rag as a Python library, you can pass configuration directly to the `HaikuRAG` client:

```python
from haiku.rag.config import AppConfig
from haiku.rag.client import HaikuRAG

# Create custom configuration
custom_config = AppConfig(
    qa={"provider": "openai", "model": "gpt-4o"},
    embeddings={"provider": "ollama", "model": "qwen3-embedding"},
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

## File Monitoring

Set directories to monitor for automatic indexing:

```yaml
monitor:
  directories:
    - /path/to/documents
    - /another_path/to/documents
```

### Filtering Monitored Files

Use gitignore-style patterns to control which files are monitored:

```yaml
monitor:
  directories:
    - /path/to/documents

  # Exclude specific files or directories
  ignore_patterns:
    - "*draft*"         # Ignore files with "draft" in the name
    - "temp/"           # Ignore temp directory
    - "**/archive/**"   # Ignore all archive directories
    - "*.backup"        # Ignore backup files

  # Only include specific files (whitelist mode)
  include_patterns:
    - "*.md"            # Only markdown files
    - "*.pdf"           # Only PDF files
    - "**/docs/**"      # Only files in docs directories
```

**How patterns work:**

1. **Extension filtering** - Only supported file types are considered
2. **Include patterns** - If specified, only matching files are included (whitelist)
3. **Ignore patterns** - Matching files are excluded (blacklist)
4. **Combining both** - Include patterns are applied first, then ignore patterns

**Common patterns:**

```yaml
# Only monitor markdown documentation, but ignore drafts
monitor:
  include_patterns:
    - "*.md"
  ignore_patterns:
    - "*draft*"
    - "*WIP*"

# Monitor all supported files except in specific directories
monitor:
  ignore_patterns:
    - "node_modules/"
    - ".git/"
    - "**/test/**"
    - "**/temp/**"
```

Patterns follow [gitignore syntax](https://git-scm.com/docs/gitignore#_pattern_format):
- `*` matches anything except `/`
- `**` matches zero or more directories
- `?` matches any single character
- `[abc]` matches any character in the set

## Embedding Providers

If you use Ollama, you can use any pulled model that supports embeddings.

### Ollama (Default)

```yaml
embeddings:
  provider: ollama
  model: mxbai-embed-large
  vector_dim: 1024
```

The Ollama base URL can be configured in your config file or via environment variable:

```yaml
providers:
  ollama:
    base_url: http://localhost:11434
```

Or via environment variable:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

If not configured, it defaults to `http://localhost:11434`.

!!! note
    You can use a `.env` file in your project directory to set environment variables like `OLLAMA_BASE_URL` and API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). These will be automatically loaded when running `haiku-rag` commands.

### VoyageAI

If you installed `haiku.rag` (full package), VoyageAI is already included. If you installed `haiku.rag-slim`, install with VoyageAI extras:

```bash
uv pip install haiku.rag-slim[voyageai]
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
```

The Ollama base URL can be configured via the `OLLAMA_BASE_URL` environment variable, config file, or defaults to `http://localhost:11434`:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

Or in your config file:

```yaml
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

If you installed `haiku.rag` (full package), MxBAI is already included. If you installed `haiku.rag-slim`, add the mxbai extra:

```bash
uv pip install haiku.rag-slim[mxbai]
```

Then configure:

```yaml
reranking:
  provider: mxbai
  model: mixedbread-ai/mxbai-rerank-base-v2
```

### Cohere

If you installed `haiku.rag` (full package), Cohere is already included. If you installed `haiku.rag-slim`, add the cohere extra:

```bash
uv pip install haiku.rag-slim[cohere]
```

Then configure:

```yaml
reranking:
  provider: cohere
  model: rerank-v3.5
```

Set your API key via environment variable:

```bash
export CO_API_KEY=your-api-key
```

### Zero Entropy

If you installed `haiku.rag` (full package), Zero Entropy is already included. If you installed `haiku.rag-slim`, add the zeroentropy extra:

```bash
uv pip install haiku.rag-slim[zeroentropy]
```

Then configure:

```yaml
reranking:
  provider: zeroentropy
  model: zerank-1  # Currently the only available model
```

Set your API key via environment variable:

```bash
export ZEROENTROPY_API_KEY=your-api-key
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

## Research Configuration

Configure the multi-agent research workflow:

```yaml
research:
  provider: ""  # Empty to use qa settings
  model: ""     # Empty to use qa model
  max_iterations: 3           # Maximum search/evaluate cycles
  confidence_threshold: 0.8   # Stop when confidence meets/exceeds this
  max_concurrency: 1          # Sub-questions searched in parallel per iteration
```

- **provider/model**: LLM provider and model for research. Leave empty to use the same settings as `qa`.
- **max_iterations**: Maximum number of search/evaluate cycles before stopping (default: 3)
- **confidence_threshold**: Stop research when evaluation confidence score meets or exceeds this threshold (default: 0.8)
- **max_concurrency**: Number of sub-questions to search in parallel during each iteration (default: 1)

The research workflow plans sub-questions, searches in parallel batches, evaluates findings, and iterates until reaching the confidence threshold or max iterations.

## AG-UI Server Configuration

Configure the AG-UI HTTP server for streaming graph execution events:

```yaml
agui:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  cors_credentials: true
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["*"]
```

Start the AG-UI server with:

```bash
haiku-rag serve --agui
```

The server exposes:
- `GET /health` - Health check endpoint
- `POST /v1/agent/stream` - Research graph streaming endpoint (Server-Sent Events)

See [Server Mode](server.md) for more details.

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

#### Database Auto-creation

haiku.rag intelligently handles database creation based on operation type:

- **Write operations** (add, add-src, delete, rebuild): Automatically create the database and required tables if they don't exist
- **Read operations** (list, get, search, ask, research): Fail with a clear error if the database doesn't exist

This prevents the common mistake where a search query accidentally creates an empty database. To initialize your database, simply add your first document using `haiku-rag add` or `haiku-rag add-src`.

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
  # Default: 86400 seconds (1 day, safe for concurrent connections)
  # Set to 0 for aggressive cleanup (removes all old versions immediately)
  vacuum_retention_seconds: 86400
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
