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

See [Configuration](configuration/index.md) for configuring providers including advanced options like vLLM.

## Requirements

- Python 3.12+
- Ollama (for default embeddings and QA)

## Pre-download Models (Optional)

You can prefetch all required runtime models before first use:

```bash
haiku-rag download-models
```

This will download:
- Docling models for document processing
- HuggingFace tokenizer models for chunking
- Any Ollama models referenced by your current configuration

## Remote Processing (Optional)

When using `haiku.rag-slim`, you can skip installing the `docling` extra and instead use [docling-serve](https://github.com/docling-project/docling-serve) for remote document processing. This is useful for:

- Keeping dependencies minimal
- Offloading heavy document processing to a dedicated service
- Production deployments with separate processing infrastructure

See [Remote processing](remote-processing.md) for setup instructions and [Document Processing](configuration/processing.md) for configuration options.

## Docker

Two Docker images are available:

### Slim Image (Minimal)

Pre-built slim image with minimal dependencies - use with external docling-serve for document processing:

```bash
docker pull ghcr.io/ggozad/haiku.rag-slim:latest
```

See `examples/docker/docker-compose.yml` for a complete setup with docling-serve.

### Full Image (Self-contained)

Build locally to include all features and document processing without docling-serve:

```bash
docker build -f docker/Dockerfile -t haiku-rag .
docker run -p 8001:8001 -v $(pwd)/data:/data haiku-rag
```

See `docker/README.md` for complete build and configuration instructions.
