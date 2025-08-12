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
If you want to use OpenAI embeddings you will need to install `haiku.rag` with the VoyageAI extras,

```bash
uv pip install haiku.rag[openai]
```

and set environment variables.

```bash
EMBEDDINGS_PROVIDER="openai"
EMBEDDINGS_MODEL="text-embedding-3-small"  # or text-embedding-3-large
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="your-api-key"
```

## Question Answering Providers

Configure which LLM provider to use for question answering.

### Ollama (Default)

```bash
QA_PROVIDER="ollama"
QA_MODEL="qwen3"
OLLAMA_BASE_URL="http://localhost:11434"
```

### OpenAI

For OpenAI QA, you need to install haiku.rag with OpenAI extras:

```bash
uv pip install haiku.rag[openai]
```

Then configure:

```bash
QA_PROVIDER="openai"
QA_MODEL="gpt-4o-mini"  # or gpt-4, gpt-3.5-turbo, etc.
OPENAI_API_KEY="your-api-key"
```

### Anthropic

For Anthropic QA, you need to install haiku.rag with Anthropic extras:

```bash
uv pip install haiku.rag[anthropic]
```

Then configure:

```bash
QA_PROVIDER="anthropic"
QA_MODEL="claude-3-5-haiku-20241022"  # or claude-3-5-sonnet-20241022, etc.
ANTHROPIC_API_KEY="your-api-key"
```

## Reranking

Reranking improves search quality by re-ordering the initial search results using specialized models. When enabled, the system retrieves more candidates (3x the requested limit) and then reranks them to return the most relevant results.

Reranking is **automatically enabled** by default using Ollama, or if you install the appropriate reranking provider package.

### Disabling Reranking

To disable reranking completely for faster searches:

```bash
RERANK_PROVIDER=""
```

### Ollama (Default)

Ollama reranking uses LLMs with structured output to rank documents by relevance:

```bash
RERANK_PROVIDER="ollama"
RERANK_MODEL="qwen3:1.7b"  # or any model that supports structured output
OLLAMA_BASE_URL="http://localhost:11434"
```

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

For Cohere reranking, install with Cohere extras:

```bash
uv pip install haiku.rag[cohere]
```

Then configure:

```bash
RERANK_PROVIDER="cohere"
RERANK_MODEL="rerank-v3.5"
COHERE_API_KEY="your-api-key"
```

## Other Settings

### Database and Storage

```bash
# Default data directory (where SQLite database is stored)
DEFAULT_DATA_DIR="/path/to/data"
```

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
