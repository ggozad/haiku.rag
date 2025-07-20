# Installation

## Basic Installation

```bash
uv pip install haiku.rag
```

By default, Ollama (with the `mxbai-embed-large` model) is used for embeddings.

## Provider-Specific Installation

For other embedding providers, install with extras:

### VoyageAI

```bash
uv pip install haiku.rag[voyageai]
```

### OpenAI

```bash
uv pip install haiku.rag[openai]
```

### Anthropic

```bash
uv pip install haiku.rag[anthropic]
```

## Requirements

- Python 3.10+
- SQLite 3.38+
- Ollama (for default embeddings)
