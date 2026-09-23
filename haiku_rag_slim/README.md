# haiku.rag-slim

The core of [haiku.rag](https://github.com/ggozad/haiku.rag), agentic RAG on LanceDB, Pydantic AI and Docling, with every heavy dependency an optional extra.

Most users want [`haiku.rag`](https://pypi.org/project/haiku.rag/), which is this package with Docling, the VoyageAI and Cohere embedders, every reranker and the terminal UI.

## Installation

Python 3.12 or newer.

```bash
uv pip install haiku.rag-slim
uv pip install 'haiku.rag-slim[docling]'          # PDF, DOCX, PPTX, XLSX, HTML, LaTeX, email and image conversion
uv pip install 'haiku.rag-slim[docling,anthropic,cross-encoder]'
```

The core includes OpenAI and Ollama support, the MCP server and Logfire. Without `docling`, documents can be converted through docling-serve.

Extras: `docling`, `tui`, `voyageai`, `cohere`, `zeroentropy`, `cross-encoder`, `jina`, `s3`, `ingester`, and one per model provider: `anthropic`, `google`, `groq`, `mistral`, `bedrock`, `vertexai`. Ollama and any OpenAI-compatible endpoint need none. [Installation](https://ggozad.github.io/haiku.rag/installation/) lists what each provides.

## Documentation

Quick start, CLI, Python API and MCP setup: [ggozad.github.io/haiku.rag](https://ggozad.github.io/haiku.rag/).
