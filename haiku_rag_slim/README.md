# haiku.rag-slim

Opinionated agentic RAG powered by LanceDB, Pydantic AI, and Docling - Core package with minimal dependencies.

`haiku.rag-slim` is the core package for users who want to install only the dependencies they need. Document processing (Docling) and the in-process rerankers are optional extras.

**For most users, we recommend installing [`haiku.rag`](https://pypi.org/project/haiku.rag/) instead**, which bundles Docling, the VoyageAI and Cohere embedders, every reranker and the terminal UI.

## Installation

**Python 3.12 or newer required**

### Minimal Installation

```bash
uv pip install haiku.rag-slim
```

Core functionality with OpenAI/Ollama support, MCP server, and Logfire observability. Document processing (docling) is optional.

### With Document Processing

```bash
uv pip install haiku.rag-slim[docling]
```

Adds PDF, DOCX, PPTX, XLSX, HTML, LaTeX, email and image conversion.

### Available Extras

`docling`, `tui`, `voyageai`, `cohere`, `zeroentropy`, `cross-encoder`, `jina`,
`s3`, `ingester`, and one per model provider: `anthropic`, `google`, `groq`,
`mistral`, `bedrock`, `vertexai`. Ollama and any OpenAI-compatible endpoint need
no extra.

What each provides, and which ones the full `haiku.rag` package already
includes: [Installation](https://ggozad.github.io/haiku.rag/installation/).

```bash
# Common combinations
uv pip install 'haiku.rag-slim[docling,anthropic,cross-encoder]'
uv pip install 'haiku.rag-slim[docling,groq]'
```

## Usage

See the main [`haiku.rag`](https://github.com/ggozad/haiku.rag) repository for:
- Quick start guide
- CLI examples
- Python API usage
- MCP server setup

## Documentation

Full documentation: https://ggozad.github.io/haiku.rag/

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Packages and extras
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - YAML configuration
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
