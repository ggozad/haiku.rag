# Installation

## Choose Your Package

**haiku.rag** is available in two packages:

### Full Package (Recommended)

```bash
uv pip install haiku.rag
```

The full package includes **all features and extras**:
- **Document processing** (Docling) - PDF, DOCX, PPTX, images, and 40+ file formats
- **All embedding providers** - Ollama, OpenAI, VoyageAI, Anthropic, vLLM
- **All rerankers** - MixedBread AI, Cohere, Zero Entropy, vLLM
- **A2A agent** - Agent-to-Agent protocol support

This is the easiest way to get started with all features enabled.

### Slim Package (Minimal Dependencies)

```bash
# Minimal installation (no document processing)
uv pip install haiku.rag-slim

# With document processing
uv pip install haiku.rag-slim[docling]

# With specific providers
uv pip install haiku.rag-slim[docling,voyageai,mxbai]
```

The slim package has minimal dependencies and lets you install only what you need:

- `docling` - PDF, DOCX, PPTX, images, and other document formats
- `voyageai` - VoyageAI embeddings
- `mxbai` - MixedBread AI reranking
- `a2a` - Agent-to-Agent protocol support
- `cohere` - Cohere reranking
- `zeroentropy` - Zero Entropy reranking

**Built-in providers** (no extras needed):
- **Ollama** (default embedding provider)
- **OpenAI** (GPT models for QA and embeddings)
- **Anthropic** (Claude models for QA)
- **vLLM** (high-performance local inference)

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

Then configure haiku.rag to use the vLLM servers. Create a `haiku.rag.yaml` file:

```yaml
embeddings:
  provider: vllm
  model: mixedbread-ai/mxbai-embed-large-v1
  vector_dim: 512

qa:
  provider: vllm
  model: Qwen/Qwen3-4B

reranking:
  provider: vllm
  model: mixedbread-ai/mxbai-rerank-base-v2

providers:
  vllm:
    embeddings_base_url: http://localhost:8000
    qa_base_url: http://localhost:8002
    rerank_base_url: http://localhost:8001
```

See [Configuration](configuration.md) for all available options.

## Requirements

- Python 3.12+
- Ollama (for default embeddings)
- vLLM server (for vLLM provider)

## Pre-download Models (Optional)

You can prefetch all required runtime models before first use:

```bash
haiku-rag download-models
```

This will download Docling models and pull any Ollama models referenced by your current configuration.

## Docker

```bash
docker pull ghcr.io/ggozad/haiku.rag:latest
```

Run the container with all services:

```bash
docker run -p 8000:8000 -p 8001:8001 -v $(pwd)/data:/data ghcr.io/ggozad/haiku.rag:latest
```

This starts the MCP server on port 8001 and A2A server on port 8000, with data persisted to `./data`.
