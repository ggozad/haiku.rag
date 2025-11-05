# haiku.rag-slim

Retrieval-Augmented Generation (RAG) library built on LanceDB - Core package with minimal dependencies.

`haiku.rag-slim` is the core package for users who want to install only the dependencies they need. Document processing (docling), rerankers, and A2A support are all optional extras.

**For most users, we recommend installing [`haiku.rag`](https://pypi.org/project/haiku.rag/) instead**, which includes all features out of the box.

## Installation

**Python 3.12 or newer required**

### Minimal Installation

```bash
uv pip install haiku.rag-slim
```

Basic functionality without document processing (docling). You can still use text input and URLs.

### With Document Processing

```bash
uv pip install haiku.rag-slim[docling]
```

Adds support for 40+ file formats including PDF, DOCX, HTML, and more.

### Available Extras

- `docling` - Document processing for PDFs, DOCX, HTML, etc.
- `voyageai` - VoyageAI embedding provider
- `mxbai` - MixedBread AI reranker
- `cohere` - Cohere reranker
- `zeroentropy` - Zero Entropy reranker
- `a2a` - Agent-to-Agent protocol support

```bash
# Multiple extras
uv pip install haiku.rag-slim[docling,voyageai,mxbai]
```

## Usage

See the main [`haiku.rag`](https://github.com/ggozad/haiku.rag) repository for:
- Quick start guide
- CLI examples
- Python API usage
- MCP server setup
- A2A agent configuration

## Documentation

Full documentation: https://ggozad.github.io/haiku.rag/

- [Installation](https://ggozad.github.io/haiku.rag/installation/) - Provider setup
- [Configuration](https://ggozad.github.io/haiku.rag/configuration/) - YAML configuration
- [CLI](https://ggozad.github.io/haiku.rag/cli/) - Command reference
- [Python API](https://ggozad.github.io/haiku.rag/python/) - Complete API docs
