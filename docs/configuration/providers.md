# Providers

haiku.rag supports multiple AI providers for embeddings, question answering, and reranking. This guide covers provider-specific configuration and setup.

!!! note
    You can use a `.env` file in your project directory to set environment variables like `OLLAMA_BASE_URL` and API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). These will be automatically loaded when running `haiku-rag` commands.

## Model Settings

Configure model behavior for the `qa` and `analysis` skills. These settings apply to any provider that supports them.

### Basic Settings

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    temperature: 0.3
    max_tokens: 500
```

**Available options:**

- **temperature**: Sampling temperature (0.0-1.0+). Defaults vary by task: 0.3 for QA and title generation, 0.0 for analysis and picture description.
  - Lower (0.0-0.3): Deterministic, focused responses
  - Medium (0.4-0.7): Balanced
  - Higher (0.8-1.0+): Creative, varied responses
- **max_tokens**: Maximum tokens in response. Default: unset (provider default), except title generation (100).
- **enable_thinking**: Control reasoning behavior (see below)
- **base_url**: Custom endpoint for OpenAI-compatible servers (vLLM, LM Studio, etc.)
- **extra_body**: Raw dict forwarded to the model SDK (see [Raw Provider Pass-through](#raw-provider-pass-through))

### Thinking Control

The `enable_thinking` setting controls whether models use explicit reasoning steps before answering.

```yaml
qa:
  model:
    enable_thinking: true   # Better grounded answers
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
- **vLLM**: Models with a pydantic-ai reasoning profile (gpt-oss). Qwen3, Gemma, and similar templates ignore the OpenAI `reasoning_effort` that `enable_thinking` translates to — use [`extra_body`](#raw-provider-pass-through) to drive them.
- **LM Studio**: Models supporting reasoning (gpt-oss, etc.)

**When to use:**
- Enable for QA, complex reasoning, and mathematical problems
- Disable for speed-critical applications, title generation, and simple tasks

!!! note "vLLM-served models without a reasoning profile"
    On `provider: openai` with a custom `base_url`, `enable_thinking` only takes effect for models whose pydantic-ai profile advertises reasoning support (o-series, gpt-5, gpt-oss). For other vLLM-served models (Qwen3, Gemma family, …) the field is a silent no-op. Reach the chat template's thinking switch directly via [`extra_body`](#raw-provider-pass-through).

### Raw Provider Pass-through

The `extra_body` setting takes a dict that haiku.rag forwards verbatim to the underlying model SDK as `ModelSettings.extra_body`. Use it to reach provider-specific keys that haiku.rag does not model with a dedicated field.

**Example: disable Qwen3 thinking on vLLM:**

```yaml
qa:
  model:
    provider: openai
    name: qwen3.6-35b
    base_url: http://localhost:11430/v1
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
```

vLLM serves Qwen3 chat templates that read their thinking switch from `chat_template_kwargs.enable_thinking`. The high-level `enable_thinking` setting on the openai provider maps to vLLM's `reasoning_effort` parameter, which Qwen3 templates ignore, so the field is a no-op for this combination. `extra_body` reaches the chat template directly and disables thinking. With it off, Qwen3 returns the answer in `content` immediately instead of emitting a hidden reasoning trace first.

**Example: enable Gemma-family thinking on vLLM:**

```yaml
qa:
  model:
    provider: openai
    name: nvidia/Gemma-4-26B-A4B-NVFP4
    base_url: http://localhost:11432/v1
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
```

Same mechanism, opposite direction. Without `extra_body` the Gemma-4 chat template defaults to non-thinking and dumps a verbose answer straight into `content`. With it on, vLLM (started with `--reasoning-parser`) populates the parsed `reasoning` field and leaves `content` as the concise final answer.

**Provider support:** honored by openai, ollama, anthropic, and groq via pydantic-ai's `ModelSettings.extra_body`. Silently ignored by gemini and bedrock.

## Embedding Providers

Embedding models require three settings: `provider`, `name`, and `vector_dim`. Optionally, use `base_url` for OpenAI-compatible servers.

### Batch Size

`embeddings.batch_size` (default `512`) sets how many text chunks are sent per `/v1/embeddings` call during ingest. Lower it if your provider caps total tokens per request. Picture embeddings are always sent one image per call and are unaffected.

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

### Cohere

Cohere embeddings are available via pydantic-ai:

```yaml
embeddings:
  model:
    provider: cohere
    name: embed-v4.0
    vector_dim: 1024
```

Set your API key via environment variable:

```bash
export CO_API_KEY=your-api-key
```

### SentenceTransformers

For local embeddings using HuggingFace models:

```yaml
embeddings:
  model:
    provider: sentence-transformers
    name: all-MiniLM-L6-v2
    vector_dim: 384
```

### OpenAI-Compatible Servers (vLLM, LM Studio, etc.)

For local inference servers with OpenAI-compatible APIs, use the `openai` provider with a custom `base_url`:

```yaml
# vLLM example
embeddings:
  model:
    provider: openai
    name: mixedbread-ai/mxbai-embed-large-v1
    vector_dim: 512
    base_url: http://localhost:8000/v1

# LM Studio example
embeddings:
  model:
    provider: openai
    name: text-embedding-qwen3-embedding-4b
    vector_dim: 2560
    base_url: http://localhost:1234/v1
```

**Note:** The `base_url` must include the `/v1` path for OpenAI-compatible endpoints.

### vLLM (multimodal)

For cross-modal retrieval (text and pictures share a single vector space), use the dedicated `vllm` provider against a vLLM server hosting a multimodal embedding model:

```yaml
embeddings:
  model:
    provider: vllm
    name: Qwen/Qwen3-VL-Embedding-8B
    vector_dim: 4096
    base_url: http://localhost:8000/v1
```

Tested with `Qwen/Qwen3-VL-Embedding-8B` (4096-dim) and `jinaai/jina-embeddings-v4` (2048-dim). Run vLLM separately. haiku.rag adds no Python ML dependencies for this path. Text inputs use the standard OpenAI `input` field. Image inputs use vLLM's `messages`-with-`image_url` superset, transparently to the caller.

Picture chunks for retrieval are emitted at ingest under any embedder reporting `supports_images=True`. See [Picture Handling](processing.md#picture-handling).

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

### OpenAI-Compatible Servers (vLLM, LM Studio, etc.)

For local inference servers with OpenAI-compatible APIs, use the `openai` provider with a custom `base_url`:

```yaml
# vLLM example
qa:
  model:
    provider: openai
    name: Qwen/Qwen3-4B
    base_url: http://localhost:8002/v1

# LM Studio example
qa:
  model:
    provider: openai
    name: gpt-oss-20b
    base_url: http://localhost:1234/v1
    enable_thinking: false
```

**Note:** The server must be running with a model that supports tool calling. The `base_url` must include the `/v1` path.

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
    base_url: http://localhost:8001
```

**Note:** vLLM reranking uses the `/v1/rerank` API endpoint. You need to run a vLLM server separately with a reranking model loaded.

### Jina AI

Jina provides high-quality reranking with two deployment options: API mode and local inference.

#### API Mode

Use the Jina Reranker API for cloud-based reranking:

```yaml
reranking:
  model:
    provider: jina
    name: jina-reranker-v3
```

Set your API key via environment variable:

```bash
export JINA_API_KEY=your-api-key
```

#### Local Mode

For local inference, install the jina extra:

```bash
uv pip install haiku.rag-slim[jina]
```

Then configure:

```yaml
reranking:
  model:
    provider: jina-local
    name: jinaai/jina-reranker-v3
```

**Note:** The Jina Reranker v3 local model is licensed under CC BY-NC 4.0, which restricts commercial use. For commercial applications, use the API mode instead.

### Cross-Encoder (sentence-transformers)

Run any HuggingFace cross-encoder reranker in-process via `sentence-transformers`. No separate server required. Useful when you want a specific model (BGE, Qwen3-Reranker, MS-MARCO MiniLM, etc.) without running vLLM.

Install the extra:

```bash
uv pip install haiku.rag-slim[cross-encoder]
```

Then configure with any HuggingFace model id:

```yaml
reranking:
  model:
    provider: cross-encoder
    name: BAAI/bge-reranker-v2-m3
```

Other tested models: `Qwen/Qwen3-Reranker-0.6B`, `cross-encoder/ms-marco-MiniLM-L-6-v2`. Any model exposed as a `sentence_transformers.CrossEncoder` works.
