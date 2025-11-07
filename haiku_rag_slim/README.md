# haiku.rag-slim

Retrieval-Augmented Generation (RAG) library built on LanceDB - Core package with minimal dependencies.

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

**Document Processing:**
- `docling` - PDF, DOCX, HTML, and 40+ file formats

**Embedding Providers:**
- `voyageai` - VoyageAI embeddings

**Rerankers:**
- `mxbai` - MixedBread AI
- `cohere` - Cohere
- `zeroentropy` - Zero Entropy

**Model Providers:**
- OpenAI/Ollama - included in core (OpenAI-compatible APIs)
- `anthropic` - Anthropic Claude
- `groq` - Groq
- `google` - Google Gemini
- `mistral` - Mistral AI
- `bedrock` - AWS Bedrock
- `vertexai` - Google Vertex AI


```bash
# Common combinations
uv pip install haiku.rag-slim[docling,anthropic,mxbai]
uv pip install haiku.rag-slim[docling,groq,logfire]
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
