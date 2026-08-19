# Installation

## Choose Your Package

**haiku.rag** is available in two packages:

### Full Package (Recommended)

```bash
uv pip install haiku.rag
```

The full package pulls the `docling`, `voyageai`, `cohere`, `zeroentropy`, `cross-encoder`, `jina` and `tui` extras:
- **Document processing** (Docling) - PDF, DOCX, PPTX, images, and 40+ file formats
- **Embedding providers** - VoyageAI and Cohere
- **Rerankers** - local cross-encoders, local Jina, Cohere, Zero Entropy

It does not include the `s3` or `ingester` extras:

```bash
uv pip install 'haiku.rag[ingester]'   # the haiku-ingester service
uv pip install 'haiku.rag[s3]'         # S3 and object storage
```

### Slim Package (Minimal Dependencies)

```bash
# Minimal installation (no document processing)
uv pip install haiku.rag-slim

# With document processing
uv pip install haiku.rag-slim[docling]

# With specific providers
uv pip install haiku.rag-slim[docling,voyageai,cross-encoder]
```

The slim package has minimal dependencies and lets you install only what you need:

- `docling` - PDF, DOCX, PPTX, images, and other document formats
- `voyageai` - VoyageAI embeddings
- `cross-encoder` - Local reranking via sentence-transformers
- `jina` - Local Jina reranking (`provider: jina-local`). Needs transformers and torch, which `cross-encoder` also pulls
- `cohere` - Cohere embeddings and reranking
- `zeroentropy` - Zero Entropy reranking
- `s3` - S3 and object-storage access
- `ingester` - The `haiku-ingester` service (also pulls `s3`)
- `tui` - Terminal UI for `chat` and `inspect` commands

**Built-in providers** (no extras needed):
- **Ollama** (default embedding provider)
- **OpenAI** (GPT models for QA and embeddings)
- **vLLM** and other OpenAI-compatible endpoints (embeddings, QA, reranking)
- **Jina** reranking via `provider: jina`, which calls the Jina HTTP API

Other Pydantic AI providers need their own Pydantic AI extra. For Claude models, install `pydantic-ai-slim[anthropic]`.

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

Only the slim image is published. Build the full image yourself:

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
docker run -p 8001:8001 \
  -v /path/to/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v /path/to/data:/data \
  haiku-rag
```

See `docker/README.md` for complete build and configuration instructions, including how to run the [ingester](ingester.md) service for continuous document ingestion.
