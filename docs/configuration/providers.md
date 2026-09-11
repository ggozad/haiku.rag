# Providers

haiku.rag supports multiple AI providers for embeddings, question answering, and reranking. This guide covers provider-specific configuration and setup.

!!! note
    You can use a `.env` file in your project directory to set environment variables like `OLLAMA_BASE_URL` and API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). These will be automatically loaded when running `haiku-rag` commands.

## Model Settings

Configure model behavior for the `qa` model. These settings apply to any provider that supports them.

### Basic Settings

```yaml
qa:
  model:
    provider: ollama
    name: qwen3.8
    temperature: 0.3
    max_tokens: 500
```

**Available options:**

- **temperature**: Sampling temperature (0.0-1.0+). Defaults vary by task: 0.3 for QA and title generation, 0.0 for picture description.
  - Lower (0.0-0.3): Deterministic, focused responses
  - Medium (0.4-0.7): Balanced
  - Higher (0.8-1.0+): Creative, varied responses
- **max_tokens**: Maximum tokens in response. Default: unset (provider default), except title generation (100).
- **thinking**: Control reasoning behavior (see below)
- **base_url**: Custom endpoint for OpenAI-compatible servers (vLLM, LM Studio, etc.)
- **api_key**: Key for this endpoint, overriding the provider's environment variable (see [Per-endpoint API keys](#per-endpoint-api-keys))
- **extra_body**: Raw dict forwarded to the model SDK (see [Raw Provider Pass-through](#raw-provider-pass-through))

### Per-endpoint API keys

The `openai` provider reads `OPENAI_API_KEY`, so several `openai`-compatible endpoints in one config would otherwise share a single key. Set `api_key` per model to give each its own, and keep the secret in the environment with [variable expansion](index.md#environment-variables):

```yaml
qa:
  model:
    provider: openai
    name: some-model
    base_url: https://vendor-a.example/v1
    api_key: ${VENDOR_A_KEY}

embeddings:
  model:
    provider: openai
    name: some-embedding-model
    vector_dim: 1024
    base_url: https://vendor-b.example/v1
    api_key: ${VENDOR_B_KEY}
```

`api_key` is honored on the `openai`, `ollama` and `vllm` providers, on `vllm` embedders and rerankers, and on the picture-description VLM endpoint (which otherwise falls back to `OPENAI_API_KEY` only for the public OpenAI endpoint, never for a custom `base_url`). Other providers (`anthropic`, `cohere`, `voyageai`, …) reach their vendor SDK by name and read their own environment variable; setting `api_key` there raises rather than being dropped silently.

### Thinking Control

The `thinking` setting controls whether models use explicit reasoning steps before answering, and at what effort.

```yaml
qa:
  model:
    thinking: true   # Better grounded answers
```

**Values:**
- `false`: Disable reasoning for faster responses
- `true`: Enable reasoning for complex tasks
- `minimal`, `low`, `medium`, `high`, `xhigh`: Enable reasoning at that effort level. The vocabulary is Pydantic AI's `ThinkingLevel`; each model accepts a subset of it, see the provider list below
- Not set: Use model defaults

`enable_thinking` is the former name of this setting. It still loads, as `thinking`, with a `FutureWarning`, and is removed in 0.90.0.

**How the value travels:**

- **Vendor APIs** (`openai` without a `base_url`, `anthropic`, `google`, `groq`, `bedrock`, and any provider reached by name): the value is passed as Pydantic AI's unified `thinking` setting, and Pydantic AI maps it per provider, clamped to what each model offers: `reasoning_effort` on OpenAI reasoning models, adaptive thinking or a token budget on Anthropic, `thinking_level` on Gemini 3. `false` is dropped on always-on models. See the [Pydantic AI thinking documentation](https://ai.pydantic.dev/thinking/) for the per-provider tables. Bedrock Converse does not serve the proprietary OpenAI models, so configuring one raises an error. Reach those through `provider: bedrock-mantle`.
- **Self-hosted OpenAI-compatible endpoints** (`ollama`, `vllm`, `openai` with a `base_url`, and the picture-description VLM): the value is sent as the request's `reasoning_effort` field under any model name: `false` sends `none`, `true` sends `medium`, a level is sent as written. The server decides what it means, so the accepted levels are the model's own.
- **Ollama** maps `reasoning_effort` onto its `think` option and never rejects a value: `minimal` becomes `low`, `xhigh` becomes `max`. Gemma 4, Qwen3.8 and Muse Glimmer think by default there. `gpt-oss` cannot be switched off and takes `low`, `medium`, `high`.
- **vLLM** hands `reasoning_effort` to the chat template, which may reject a value it does not know: `Inferact/Qwen3.8-27B-NVFP4` takes `low`, `medium` and `xhigh` and returns 400 for `minimal` and `high`. The Gemma 4 family reads only on and off, so every level thinks the same. A template with a switch of its own, Muse Glimmer's `reasoning_strength`, ignores the field, see [vLLM](#vllm).
- **LM Studio** is reached through `openai` with a `base_url` and receives `reasoning_effort`. Its documented chat API lists no reasoning parameter.

**When to use:**
- Enable for QA, complex reasoning, and mathematical problems
- Disable for speed-critical applications, title generation, and simple tasks

!!! note "Anthropic thinking and max_tokens"
    Anthropic requires `max_tokens` to exceed the thinking budget, and `thinking: true` requests Pydantic AI's default budget of 10000 tokens. Set `max_tokens` above 10000 on Claude models that use budget-based thinking, or leave it unset on Sonnet 4.6+ and Opus 4.6+, which use adaptive thinking instead of a budget.

### Raw Provider Pass-through

The `extra_body` setting takes a dict that haiku.rag forwards verbatim to the underlying model SDK as `ModelSettings.extra_body`. Use it to reach provider-specific keys that haiku.rag does not model with a dedicated field.

**Example: vLLM sampling parameters:**

```yaml
qa:
  model:
    provider: vllm
    name: Inferact/Qwen3.8-27B-NVFP4
    base_url: http://localhost:11439
    extra_body:
      top_p: 0.95
      top_k: 20
      min_p: 0
```

These keys are sent as top-level request fields. Of the three, ollama honors only `top_p`.

`extra_body.reasoning_effort` reaches the request the same way and overrides the value `thinking` sends. A template carrying a switch of its own takes `chat_template_kwargs`, see [vLLM](#vllm).

**Provider support:** honored by openai, ollama, anthropic, groq and vllm via pydantic-ai's `ModelSettings.extra_body`. Silently ignored by google and bedrock.

## Embedding Providers

Embedding models require three settings: `provider`, `name`, and `vector_dim`. Optionally, use `base_url` for OpenAI-compatible servers and [`api_key`](#per-endpoint-api-keys) for the key that endpoint expects.

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

**Note:** On `provider: openai` the `base_url` must include the `/v1` path. This path is text-only. For a vision-language model served by vLLM, use `provider: vllm` with `multimodal: true` (below), not `provider: openai`.

### Multimodal embedders

For cross-modal retrieval (text and pictures share a single vector space), set `embeddings.model.multimodal: true`. Capability is decided by this flag, not the provider name: each provider passes images in its own wire format, so multimodal is supported only on `vllm`, `voyageai`, and `cohere`. Setting it on any other provider raises at startup.

A model produces picture chunks at ingest only when its embedder is multimodal. Without the flag, an image-only document produces zero chunks and is not retrievable. Switching `multimodal` on or off does not change the stored embedding identity, so it raises no drift error; re-ingest or `rebuild` to add or drop picture chunks.

**vLLM** — a vLLM server hosting a multimodal embedding model. Text inputs use the standard OpenAI `input` field; image inputs use vLLM's `messages`-with-`image_url` superset. Tested with `Qwen/Qwen3-VL-Embedding-8B` (4096-dim) and `jinaai/jina-embeddings-v4` (2048-dim). Run vLLM separately; haiku.rag adds no Python ML dependencies for this path.

```yaml
embeddings:
  model:
    provider: vllm
    name: Qwen/Qwen3-VL-Embedding-8B
    vector_dim: 4096
    base_url: http://localhost:8000/v1
    multimodal: true
```

**VoyageAI** — `voyage-multimodal-3` (1024-dim) via the `voyageai` extra. Reads `VOYAGE_API_KEY` from the environment.

```yaml
embeddings:
  model:
    provider: voyageai
    name: voyage-multimodal-3
    vector_dim: 1024
    multimodal: true
```

**Cohere** — `embed-v4.0` (configurable `vector_dim`, e.g. 1536) via the `cohere` extra. Reads `CO_API_KEY` from the environment.

```yaml
embeddings:
  model:
    provider: cohere
    name: embed-v4.0
    vector_dim: 1536
    multimodal: true
```

A text-only model served by vLLM uses `provider: vllm` without the flag (or `provider: openai` with a `base_url`).

Picture chunks for retrieval are emitted at ingest under any multimodal embedder. See [Picture Handling](processing.md#picture-handling).

## Question Answering Providers

Configure which LLM provider to use for question answering. Any provider and model supported by [Pydantic AI](https://ai.pydantic.dev/models/) can be used.

### Ollama (Default)

```yaml
qa:
  model:
    provider: ollama
    name: qwen3.8
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

`provider: vllm` is the spelling for a vLLM-served chat model, and `provider: openai` with a `base_url` also works. `base_url` is accepted with or without the `/v1` path:

```yaml
qa:
  model:
    provider: vllm
    name: Qwen/Qwen3-4B
    base_url: http://localhost:8002
```

The provider brings its own model profile, which merges leading system messages
(some chat templates reject more than one) and picks tool-schema behaviour per
family, inferred from the model name. `thinking` does not depend on that
inference: it is sent as `reasoning_effort` under any served name.

The accepted levels are the model's, and the server decides:
`Inferact/Qwen3.8-27B-NVFP4` takes `low`, `medium` and `xhigh` and returns 400
for `minimal` and `high`. The Gemma 4 family reads only on and off. Write the
value the model accepts:

```yaml
qa:
  model:
    provider: vllm
    name: Inferact/Qwen3.8-27B-NVFP4
    base_url: http://localhost:11439
    thinking: xhigh
```

A template with a switch of its own takes `chat_template_kwargs` instead, as
Muse Glimmer does. It accepts `reasoning_effort` and ignores it, so `thinking`
has no effect on it under vLLM:

```yaml
qa:
  model:
    provider: vllm
    name: RedHatAI/Muse-Glimmer-30B-NVFP4
    base_url: http://localhost:11450
    extra_body:
      chat_template_kwargs:
        reasoning_strength: high
```

`chat_template_kwargs.enable_thinking` takes precedence over a derived
`reasoning_effort`. vLLM (0.28.0) derives `enable_thinking` from `reasoning_effort`
only when the request does not set it, so on a template that reads only that
switch, Gemma 4 among them, the template kwarg decides in both directions. Both
values still travel, so leave `thinking` unset when you set the template
switch.

`provider: vllm` under `embeddings.model` and `reranking.model` is a different
implementation: haiku.rag's own client for vLLM's native multimodal endpoints.
Setting it in one place says nothing about the other.

### Other OpenAI-Compatible Servers (LM Studio, sglang, etc.)

For other local inference servers with OpenAI-compatible APIs, use the `openai` provider with a custom `base_url`:

```yaml
qa:
  model:
    provider: openai
    name: gpt-oss-20b
    base_url: http://localhost:1234/v1
    thinking: false
```

**Note:** The server must be running with a model that supports tool calling. On the `openai` provider the `base_url` must include the `/v1` path.

### Other Providers

Any provider supported by Pydantic AI can be used. Examples:

```yaml
# Google Gemini
qa:
  model:
    provider: google
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

Reranking is **disabled by default** for faster searches: there is no `reranking.model`. Enable it by configuring one of the providers below, and disable it again by removing the section or setting `model: null`.

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

For local reranking with a dedicated reranking model:

```yaml
reranking:
  model:
    provider: vllm
    name: Qwen/Qwen3-Reranker-4B
    base_url: http://localhost:8001/v1
```

**Note:** vLLM reranking posts to the `/v1/rerank` endpoint. As with the embedder, `base_url` may be written with or without the `/v1` path. You need to run a vLLM server separately with a reranking model loaded.

#### Multimodal reranking

When serving a vision reranker (for example `nvidia/llama-nemotron-rerank-vl-1b-v2`), set `multimodal: true` to score picture chunks by their image bytes in addition to their description text:

```yaml
reranking:
  multimodal: true
  model:
    provider: vllm
    name: nvidia/llama-nemotron-rerank-vl-1b-v2
    base_url: http://localhost:8001/v1
```

Picture chunks are sent as image documents (base64 data URIs) alongside plain text documents in the same rerank request. The flag is supported on the vllm provider only, and the served model must accept multimodal inputs.

### Jina AI

Jina reranking has two deployment options: API mode and local inference.

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
    name: Qwen/Qwen3-Reranker-0.6B
```

Other tested models: `BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`. Any model exposed as a `sentence_transformers.CrossEncoder` works.
