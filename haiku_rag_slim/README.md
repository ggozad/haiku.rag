# haiku.rag-slim

Opinionated agentic RAG powered by LanceDB, Pydantic AI, and Docling - Core package with minimal dependencies.

`haiku.rag-slim` is the core package for users who want to install only the dependencies they need. Document processing (docling), and reranker support are all optional extras.

**For most users, we recommend installing [`haiku.rag`](https://pypi.org/project/haiku.rag/) instead**, which includes all features out of the box.

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

Adds support for 40+ file formats including PDF, DOCX, HTML, and more.

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

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Provider setup
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - YAML configuration
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
