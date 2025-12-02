# Providers

haiku.rag supports multiple AI providers for embeddings, question answering, and reranking. This guide covers provider-specific configuration and setup.

!!! note
    You can use a `.env` file in your project directory to set environment variables like `OLLAMA_BASE_URL` and API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). These will be automatically loaded when running `haiku-rag` commands.

## Model Settings

Configure model behavior for `qa` and `research` workflows. These settings apply to any provider that supports them.

### Basic Settings

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    temperature: 0.7
    max_tokens: 500
```

**Available options:**

- **temperature**: Sampling temperature (0.0-1.0+)
  - Lower (0.0-0.3): Deterministic, focused responses
  - Medium (0.4-0.7): Balanced
  - Higher (0.8-1.0+): Creative, varied responses
- **max_tokens**: Maximum tokens in response
- **enable_thinking**: Control reasoning behavior (see below)

### Thinking Control

The `enable_thinking` setting controls whether models use explicit reasoning steps before answering.

```yaml
qa:
  model:
    enable_thinking: false  # Faster responses

research:
  model:
    enable_thinking: true   # Deeper reasoning
```

**Values:**
- `false`: Disable reasoning for faster responses
- `true`: Enable reasoning for complex tasks
- Not set: Use model defaults

**Provider support:**

See the [Pydantic AI thinking documentation](https://ai.pydantic.dev/thinking/) for detailed provider support. haiku.rag supports thinking control for:

- **OpenAI**: Reasoning models (o1, o3, gpt-oss)
- **Anthropic**: All Claude models
- **Google**: Gemini models with thinking support
- **Groq**: Models with reasoning capabilities
- **Bedrock**: Claude, OpenAI, and Qwen models
- **Ollama**: Models supporting reasoning (gpt-oss, etc.)
- **vLLM**: Models supporting reasoning (gpt-oss, etc.)
- **LM Studio**: Models supporting reasoning (gpt-oss, etc.)

**When to use:**
- Disable for simple queries, RAG workflows, speed-critical applications
- Enable for complex reasoning, mathematical problems, research tasks

## Embedding Providers

If you use Ollama, you can use any pulled model that supports embeddings.

### Ollama (Default)

```yaml
embeddings:
  model:
    provider: ollama
    name: mxbai-embed-large
    vector_dim: 1024
```

The Ollama base URL can be configured in your config file or via environment variable:

```yaml
providers:
  ollama:
    base_url: http://localhost:11434
```

Or via environment variable:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

If not configured, it defaults to `http://localhost:11434`.

### VoyageAI

If you installed `haiku.rag` (full package), VoyageAI is already included. If you installed `haiku.rag-slim`, install with VoyageAI extras:

```bash
uv pip install haiku.rag-slim[voyageai]
```

```yaml
embeddings:
  model:
    provider: voyageai
    name: voyage-3.5
    vector_dim: 1024
```

Set your API key via environment variable:

```bash
export VOYAGE_API_KEY=your-api-key
```

### OpenAI

OpenAI embeddings are included in the default installation:

```yaml
embeddings:
  model:
    provider: openai
    name: text-embedding-3-small  # or text-embedding-3-large
    vector_dim: 1536
```

Set your API key via environment variable:

```bash
export OPENAI_API_KEY=your-api-key
```

### vLLM

For high-performance local inference, you can use vLLM to serve embedding models with OpenAI-compatible APIs:

```yaml
embeddings:
  model:
    provider: vllm
    name: mixedbread-ai/mxbai-embed-large-v1
    vector_dim: 512

providers:
  vllm:
    embeddings_base_url: http://localhost:8000
```

**Note:** You need to run a vLLM server separately with an embedding model loaded.

### LM Studio

[LM Studio](https://lmstudio.ai/) provides a local OpenAI-compatible API server for running models:

```yaml
embeddings:
  model:
    provider: lm_studio
    name: text-embedding-qwen3-embedding-4b
    vector_dim: 2560

providers:
  lm_studio:
    base_url: http://localhost:1234
```

**Note:** LM Studio must be running with an embedding model loaded. The default URL is `http://localhost:1234`.

## Question Answering Providers

Configure which LLM provider to use for question answering. Any provider and model supported by [Pydantic AI](https://ai.pydantic.dev/models/) can be used.

### Ollama (Default)

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
```

The Ollama base URL can be configured via the `OLLAMA_BASE_URL` environment variable, config file, or defaults to `http://localhost:11434`:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

Or in your config file:

```yaml
providers:
  ollama:
    base_url: http://localhost:11434
```

### OpenAI

OpenAI QA is included in the default installation:

```yaml
qa:
  model:
    provider: openai
    name: gpt-4o-mini  # or gpt-4, gpt-3.5-turbo, etc.
```

Set your API key via environment variable:

```bash
export OPENAI_API_KEY=your-api-key
```

### Anthropic

Anthropic QA is included in the default installation:

```yaml
qa:
  model:
    provider: anthropic
    name: claude-3-5-haiku-20241022  # or claude-3-5-sonnet-20241022, etc.
```

Set your API key via environment variable:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

### vLLM

For high-performance local inference:

```yaml
qa:
  model:
    provider: vllm
    name: Qwen/Qwen3-4B  # Any model with tool support in vLLM

providers:
  vllm:
    qa_base_url: http://localhost:8002
```

**Note:** You need to run a vLLM server separately with a model that supports tool calling loaded. Consult the specific model's documentation for proper vLLM serving configuration.

### LM Studio

Use LM Studio for local question answering and research:

```yaml
qa:
  model:
    provider: lm_studio
    name: openai/gpt-oss-20b
    enable_thinking: false

research:
  model:
    provider: lm_studio
    name: openai/gpt-oss-20b

providers:
  lm_studio:
    base_url: http://localhost:1234
```

**Note:** LM Studio must be running with a chat model that supports tool calling loaded.

### Other Providers

Any provider supported by Pydantic AI can be used. Examples:

```yaml
# Google Gemini
qa:
  model:
    provider: gemini
    name: gemini-1.5-flash

# Groq
qa:
  model:
    provider: groq
    name: llama-3.3-70b-versatile

# Mistral
qa:
  model:
    provider: mistral
    name: mistral-small-latest
```

See the [Pydantic AI documentation](https://ai.pydantic.dev/models/) for the complete list of supported providers and models.

## Reranking Providers

Reranking improves search quality by re-ordering the initial search results using specialized models. When enabled, the system retrieves more candidates (10x the requested limit) and then reranks them to return the most relevant results.

Reranking is **disabled by default** (`provider: ""`) for faster searches. You can enable it by configuring one of the providers below.

### MixedBread AI

If you installed `haiku.rag` (full package), MxBAI is already included. If you installed `haiku.rag-slim`, add the mxbai extra:

```bash
uv pip install haiku.rag-slim[mxbai]
```

Then configure:

```yaml
reranking:
  model:
    provider: mxbai
    name: mixedbread-ai/mxbai-rerank-base-v2
```

### Cohere

If you installed `haiku.rag` (full package), Cohere is already included. If you installed `haiku.rag-slim`, add the cohere extra:

```bash
uv pip install haiku.rag-slim[cohere]
```

Then configure:

```yaml
reranking:
  model:
    provider: cohere
    name: rerank-v3.5
```

Set your API key via environment variable:

```bash
export CO_API_KEY=your-api-key
```

### Zero Entropy

If you installed `haiku.rag` (full package), Zero Entropy is already included. If you installed `haiku.rag-slim`, add the zeroentropy extra:

```bash
uv pip install haiku.rag-slim[zeroentropy]
```

Then configure:

```yaml
reranking:
  model:
    provider: zeroentropy
    name: zerank-1  # Currently the only available model
```

Set your API key via environment variable:

```bash
export ZEROENTROPY_API_KEY=your-api-key
```

### vLLM

For high-performance local reranking using dedicated reranking models:

```yaml
reranking:
  model:
    provider: vllm
    name: mixedbread-ai/mxbai-rerank-base-v2

providers:
  vllm:
    rerank_base_url: http://localhost:8001
```

**Note:** vLLM reranking uses the `/rerank` API endpoint. You need to run a vLLM server separately with a reranking model loaded. Consult the specific model's documentation for proper vLLM serving configuration.
