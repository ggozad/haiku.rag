# Configuration

Configuration is done through the use of environment variables.

!!! note
    If you create a db with certain settings and later change them, `haiku.rag` will detect incompatibilities (for example, if you change embedding provider) and will exit. You can **rebuild** the database to apply the new settings, see [Rebuild Database](./cli.md#rebuild-database).

## File Monitoring

Set directories to monitor for automatic indexing:

```bash
# Monitor single directory
MONITOR_DIRECTORIES="/path/to/documents"

# Monitor multiple directories
MONITOR_DIRECTORIES="/path/to/documents,/another_path/to/documents"
```

## Embedding Providers

If you use Ollama, you can use any pulled model that supports embeddings.

### Ollama (Default)

```bash
EMBEDDINGS_PROVIDER="ollama"
EMBEDDINGS_MODEL="mxbai-embed-large"
EMBEDDINGS_VECTOR_DIM=1024
```

### VoyageAI
If you want to use VoyageAI embeddings you will need to install `haiku.rag` with the VoyageAI extras,

```bash
uv pip install haiku.rag[voyageai]
```

```bash
EMBEDDINGS_PROVIDER="voyageai"
EMBEDDINGS_MODEL="voyage-3.5"
EMBEDDINGS_VECTOR_DIM=1024
VOYAGE_API_KEY="your-api-key"
```

### OpenAI
OpenAI embeddings are included in the default installation. Simply set environment variables:

```bash
EMBEDDINGS_PROVIDER="openai"
EMBEDDINGS_MODEL="text-embedding-3-small"  # or text-embedding-3-large
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="your-api-key"
```

### vLLM
For high-performance local inference, you can use vLLM to serve embedding models with OpenAI-compatible APIs:

```bash
EMBEDDINGS_PROVIDER="vllm"
EMBEDDINGS_MODEL="mixedbread-ai/mxbai-embed-large-v1"  # Any embedding model supported by vLLM
EMBEDDINGS_VECTOR_DIM=512  # Dimension depends on the model
VLLM_EMBEDDINGS_BASE_URL="http://localhost:8000"  # vLLM server URL
```

**Note:** You need to run a vLLM server separately with an embedding model loaded.

## Question Answering Providers

Configure which LLM provider to use for question answering. Any provider and model supported by [Pydantic AI](https://ai.pydantic.dev/models/) can be used.

### Ollama (Default)

```bash
QA_PROVIDER="ollama"
QA_MODEL="qwen3"
OLLAMA_BASE_URL="http://localhost:11434"
```

### OpenAI

OpenAI QA is included in the default installation. Simply configure:

```bash
QA_PROVIDER="openai"
QA_MODEL="gpt-4o-mini"  # or gpt-4, gpt-3.5-turbo, etc.
OPENAI_API_KEY="your-api-key"
```

### Anthropic

Anthropic QA is included in the default installation. Simply configure:

```bash
QA_PROVIDER="anthropic"
QA_MODEL="claude-3-5-haiku-20241022"  # or claude-3-5-sonnet-20241022, etc.
ANTHROPIC_API_KEY="your-api-key"
```

### vLLM

For high-performance local inference, you can use vLLM to serve models with OpenAI-compatible APIs:

```bash
QA_PROVIDER="vllm"
QA_MODEL="Qwen/Qwen3-4B"  # Any model with tool support in vLLM
VLLM_QA_BASE_URL="http://localhost:8002"  # vLLM server URL
```

**Note:** You need to run a vLLM server separately with a model that supports tool calling loaded. Consult the specific model's documentation for proper vLLM serving configuration.

### Other Providers

Any provider supported by Pydantic AI can be used. Examples include:

```bash
# Google Gemini
QA_PROVIDER="gemini"
QA_MODEL="gemini-1.5-flash"

# Groq
QA_PROVIDER="groq"
QA_MODEL="llama-3.3-70b-versatile"

# Mistral
QA_PROVIDER="mistral"
QA_MODEL="mistral-small-latest"
```

See the [Pydantic AI documentation](https://ai.pydantic.dev/models/) for the complete list of supported providers and models.

## Reranking

Reranking improves search quality by re-ordering the initial search results using specialized models. When enabled, the system retrieves more candidates (3x the requested limit) and then reranks them to return the most relevant results.

Reranking is **disabled by default** (`RERANK_PROVIDER=""`) for faster searches. You can enable it by configuring one of the providers below.

### MixedBread AI

For MxBAI reranking, install with mxbai extras:

```bash
uv pip install haiku.rag[mxbai]
```

Then configure:

```bash
RERANK_PROVIDER="mxbai"
RERANK_MODEL="mixedbread-ai/mxbai-rerank-base-v2"
```

### Cohere

Cohere reranking is included in the default installation. Simply configure:

```bash
RERANK_PROVIDER="cohere"
RERANK_MODEL="rerank-v3.5"
COHERE_API_KEY="your-api-key"
```

### vLLM

For high-performance local reranking using dedicated reranking models:

```bash
RERANK_PROVIDER="vllm"
RERANK_MODEL="mixedbread-ai/mxbai-rerank-base-v2"  # Any reranking model supported by vLLM
VLLM_RERANK_BASE_URL="http://localhost:8001"  # vLLM server URL
```

**Note:** vLLM reranking uses the `/rerank` API endpoint. You need to run a vLLM server separately with a reranking model loaded. Consult the specific model's documentation for proper vLLM serving configuration.

## Other Settings

### Database and Storage

By default, `haiku.rag` uses a local LanceDB database:

```bash
# Default data directory (where local LanceDB is stored)
DEFAULT_DATA_DIR="/path/to/data"
```

For remote storage, use the `LANCEDB_URI` setting with various backends:

```bash
# LanceDB Cloud
LANCEDB_URI="db://your-database-name"
LANCEDB_API_KEY="your-api-key"
LANCEDB_REGION="us-west-2"  # optional

# Amazon S3
LANCEDB_URI="s3://my-bucket/my-table"
# Use AWS credentials or IAM roles

# Azure Blob Storage
LANCEDB_URI="az://my-container/my-table"
# Use Azure credentials

# Google Cloud Storage
LANCEDB_URI="gs://my-bucket/my-table"
# Use GCP credentials

# HDFS
LANCEDB_URI="hdfs://namenode:port/path/to/table"
```

Authentication is handled through standard cloud provider credentials (AWS CLI, Azure CLI, gcloud, etc.) or by setting `LANCEDB_API_KEY` for LanceDB Cloud.

**Note:** Table optimization is automatically handled by LanceDB Cloud (`db://` URIs) and is disabled for better performance. For object storage backends (S3, Azure, GCS), optimization is still performed locally.

#### Disable database auto-creation

By default, haiku.rag creates the local LanceDB directory and required tables on first use. To prevent accidental database creation and fail fast if a database hasn’t been set up yet, set:

```bash
DISABLE_DB_AUTOCREATE=true
```

When enabled, for local paths, haiku.rag errors if the LanceDB directory does not exist, and it will not create parent directories.

### Document Processing

```bash
# Chunk size for document processing
CHUNK_SIZE=256

# Number of adjacent chunks to include before/after retrieved chunks for context
# 0 = no expansion (default), 1 = include 1 chunk before and after, etc.
# When expanded chunks overlap or are adjacent, they are automatically merged
# into single chunks with continuous content to eliminate duplication
CONTEXT_CHUNK_RADIUS=0
```

#### Markdown Preprocessor

Optionally preprocess Markdown before chunking by pointing to a callable that receives and returns Markdown text. This is useful for normalizing content, stripping boilerplate, or applying custom transformations before chunk boundaries are computed.

```bash
# A callable path in one of these formats:
# - package.module:func
# - package.module.func
# - /abs/or/relative/path/to/file.py:func
MARKDOWN_PREPROCESSOR="my_pkg.preprocess:clean_md"
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
