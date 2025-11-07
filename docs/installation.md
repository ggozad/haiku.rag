# Installation

## Choose Your Package

**haiku.rag** is available in two packages:

### Full Package (Recommended)

```bash
uv pip install haiku.rag
```

The full package includes **all features and extras**:
- **Document processing** (Docling) - PDF, DOCX, PPTX, images, and 40+ file formats
- **All embedding providers** - VoyageAI
- **All rerankers** - MixedBread AI, Cohere, Zero Entropy

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
- `cohere` - Cohere reranking
- `zeroentropy` - Zero Entropy reranking

**Built-in providers** (no extras needed):
- **Ollama** (default embedding provider)
- **OpenAI** (GPT models for QA and embeddings)
- **Anthropic** (Claude models for QA)

See [Configuration](configuration.md) for configuring providers including advanced options like vLLM.

## Requirements

- Python 3.12+
- Ollama (for default embeddings and QA)

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

This starts the MCP server on port 8001, with data persisted to `./data`.
