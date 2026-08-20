# Installation

## Choose Your Package

**haiku.rag** is available in two packages:

### Full Package (Recommended)

```bash
uv pip install haiku.rag
```

The full package pulls the `docling`, `voyageai`, `cohere`, `zeroentropy`,
`cross-encoder`, `jina` and `tui` extras. It does not include `s3` or `ingester`:

```bash
uv pip install 'haiku.rag[ingester]'   # the haiku-ingester service
uv pip install 'haiku.rag[s3]'         # S3 and object storage
```

### Slim Package (Minimal Dependencies)

```bash
uv pip install haiku.rag-slim
uv pip install 'haiku.rag-slim[docling]'
uv pip install 'haiku.rag-slim[docling,voyageai,cross-encoder]'
```

### Extras

Every extra `haiku.rag-slim` defines. The right-hand column marks the ones the
full `haiku.rag` package already includes.

| Extra | Provides | In `haiku.rag` |
|---|---|---|
| `docling` | PDF, DOCX, PPTX, images and 40+ formats, converted locally | yes |
| `tui` | Terminal UI for `chat` and `inspect` | yes |
| `voyageai` | VoyageAI embeddings | yes |
| `cohere` | Cohere embeddings and reranking | yes |
| `zeroentropy` | Zero Entropy reranking | yes |
| `cross-encoder` | Local reranking via sentence-transformers | yes |
| `jina` | Local Jina reranking (`provider: jina-local`) | yes |
| `s3` | S3 and object-storage access | no |
| `ingester` | The `haiku-ingester` service (also pulls `s3`) | no |
| `anthropic` | Anthropic Claude models | no |
| `google` | Google Gemini models | no |
| `groq` | Groq models | no |
| `mistral` | Mistral models | no |
| `bedrock` | AWS Bedrock models | no |
| `vertexai` | Google Vertex AI models | no |

Ollama and any OpenAI-compatible endpoint work with no extra at all.

**Built-in providers** (no extras needed):
- **Ollama** (default embedding provider)
- **OpenAI** (GPT models for QA and embeddings)
- **vLLM** and other OpenAI-compatible endpoints (embeddings, QA, reranking)
- **Jina** reranking via `provider: jina`, which calls the Jina HTTP API

Other providers come from the extras above, which pull the matching Pydantic AI extra. For Claude models, `uv pip install 'haiku.rag-slim[anthropic]'`.

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
