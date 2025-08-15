# Installation

## Basic Installation

```bash
uv pip install haiku.rag
```

This includes support for:
- **Ollama** (default embedding provider using `mxbai-embed-large`)
- **OpenAI** (GPT models for QA and embeddings)
- **Anthropic** (Claude models for QA)
- **Cohere** (reranking models)

## Provider-Specific Installation

For additional embedding providers, install with extras:

### VoyageAI

```bash
uv pip install haiku.rag[voyageai]
```

### MixedBread AI Reranking

```bash
uv pip install haiku.rag[mxbai]
```

## Requirements

- Python 3.10+
- SQLite 3.38+
- Ollama (for default embeddings)
