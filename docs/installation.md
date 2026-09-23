# Installation

haiku.rag needs Python 3.12 or newer. The default configuration runs its embedding and answering models through [Ollama](https://ollama.com/).

## Packages

haiku.rag ships as two packages:

```bash
uv pip install haiku.rag        # full
uv pip install haiku.rag-slim   # core, extras chosen by you
```

`haiku.rag` is `haiku.rag-slim` with the `docling`, `voyageai`, `cohere`, `zeroentropy`, `cross-encoder`, `jina` and `tui` extras. It defines four extras of its own, `tui`, `s3`, `cross-encoder` and `ingester`:

```bash
uv pip install 'haiku.rag[ingester]'   # the haiku-ingester service
uv pip install 'haiku.rag[s3]'         # S3 and object storage
```

The model-provider extras exist only on `haiku.rag-slim`. With the full package, install them beside it:

```bash
uv pip install haiku.rag 'haiku.rag-slim[anthropic]'
```

## Extras

Every extra `haiku.rag-slim` defines:

| Extra | Provides | In `haiku.rag` |
|---|---|---|
| `docling` | PDF, DOCX, PPTX, XLSX, HTML, LaTeX, email and images, converted locally | yes |
| `tui` | Terminal UI for `chat` and `inspect` | yes |
| `voyageai` | VoyageAI embeddings | yes |
| `cohere` | Cohere embeddings and reranking | yes |
| `zeroentropy` | Zero Entropy reranking | yes |
| `cross-encoder` | Local reranking and embeddings via sentence-transformers | yes |
| `jina` | Local Jina reranking (`provider: jina-local`) | yes |
| `s3` | S3 and object-storage access | no |
| `ingester` | The `haiku-ingester` service (also pulls `s3`) | no |
| `anthropic` | Anthropic Claude models | no |
| `google` | Google Gemini models | no |
| `groq` | Groq models | no |
| `mistral` | Mistral models | no |
| `bedrock` | AWS Bedrock models | no |
| `vertexai` | Google Vertex AI models | no |

These providers need no extra: Ollama, OpenAI and any OpenAI-compatible server (vLLM, LM Studio, sglang), OpenRouter, and Jina reranking through its HTTP API (`provider: jina`). [Providers](configuration/providers.md) covers configuring each.

Without the `docling` extra, `haiku.rag-slim` can convert through [docling-serve](remote-processing.md) instead.

## Pre-download models

```bash
haiku-rag download-models
```

fetches, ahead of first use:

- Docling models for document processing
- The HuggingFace tokenizer for chunking
- A sentence-transformers embedder, and a cross-encoder or local Jina reranker, when configured
- Every Ollama model the configuration references: embeddings, QA, reranking, title generation and picture description

## Docker

The slim image is published, with the `ingester` extra and without `docling`, for use with docling-serve:

```bash
docker pull ghcr.io/ggozad/haiku.rag-slim:latest
```

`examples/docker/docker-compose.yml` runs it with docling-serve, the ingester and the MCP server.

The full image, which converts in-process, is built locally:

```bash
docker build -f docker/Dockerfile -t haiku-rag .
docker run -p 8001:8001 \
  -v /path/to/haiku.rag.yaml:/app/haiku.rag.yaml \
  -v /path/to/data:/data \
  haiku-rag
```

Both images run the read-only MCP server on port 8001 by default. The mounted `haiku.rag.yaml` must set `storage.data_dir: /data`, or the database is written inside the container rather than to the volume. `docker/README.md` covers running the [ingester](ingester.md) from the same image.
