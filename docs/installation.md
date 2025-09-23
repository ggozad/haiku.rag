# Installation

## Basic Installation

```bash
uv pip install haiku.rag
```

This includes support for:
- **Ollama** (default embedding provider using `mxbai-embed-large`)
- **OpenAI** (GPT models for QA and embeddings)
- **Anthropic** (Claude models for QA)
- **Cohere** (reranking models)
- **vLLM** (high-performance local inference for embeddings, QA, and reranking)

## Provider-Specific Installation

For additional embedding providers, install with extras:

### VoyageAI

```bash
uv pip install haiku.rag[voyageai]
```

### MixedBread AI Reranking

```bash
uv pip install haiku.rag[mxbai]
```

### vLLM Setup

vLLM requires no additional installation - it works with the base haiku.rag package. However, you need to run vLLM servers separately:

```bash
# Install vLLM
pip install vllm

# Serve an embedding model
vllm serve mixedbread-ai/mxbai-embed-large-v1 --port 8000

# Serve a model for QA (requires tool calling support)
vllm serve Qwen/Qwen3-4B --port 8002 --enable-auto-tool-choice --tool-call-parser hermes

# Serve a model for reranking
vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}' --port 8001
```

Then configure haiku.rag to use the vLLM servers:

```bash
# Embeddings
EMBEDDINGS_PROVIDER="vllm"
EMBEDDINGS_MODEL="mixedbread-ai/mxbai-embed-large-v1"
EMBEDDINGS_VECTOR_DIM=512
VLLM_EMBEDDINGS_BASE_URL="http://localhost:8000"

# QA (optional)
QA_PROVIDER="vllm"
QA_MODEL="Qwen/Qwen3-4B"
VLLM_QA_BASE_URL="http://localhost:8002"

# Reranking (optional)
RERANK_PROVIDER="vllm"
RERANK_MODEL="mixedbread-ai/mxbai-rerank-base-v2"
VLLM_RERANK_BASE_URL="http://localhost:8001"
```

## Requirements

- Python 3.10+
- Ollama (for default embeddings)
- vLLM server (for vLLM provider)

## Pre-download Models (Optional)

You can prefetch all required runtime models before first use:

```bash
haiku-rag download-models
```

This will download Docling models and pull any Ollama models referenced by your current configuration.
