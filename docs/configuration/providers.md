# Providers

haiku.rag uses three kinds of model: an embedder, a question-answering model, and an optional reranker. Each is configured with a `provider` and a `name`.

## Summary

| Provider | QA | Embeddings | Reranking | Extra on `haiku.rag-slim` | Key |
|---|---|---|---|---|---|
| `ollama` | yes | yes | | none | |
| `openai` | yes | yes | | none | `OPENAI_API_KEY` |
| `vllm` | yes | yes, multimodal | yes, multimodal | none | |
| `openrouter` | yes | yes, multimodal | yes, multimodal | none | `OPENROUTER_API_KEY` |
| `anthropic` | yes | | | `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | yes | | | `google` | `GOOGLE_API_KEY` |
| `groq` | yes | | | `groq` | `GROQ_API_KEY` |
| `mistral` | yes | | | `mistral` | `MISTRAL_API_KEY` |
| `bedrock` | yes | | | `bedrock` | AWS credentials |
| `voyageai` | | yes, multimodal | | `voyageai` (in `haiku.rag`) | `VOYAGE_API_KEY` |
| `cohere` | | yes, multimodal | yes | `cohere` (in `haiku.rag`) | `CO_API_KEY` |
| `sentence-transformers` | | yes | | `cross-encoder` (in `haiku.rag`) | |
| `cross-encoder` | | | yes | `cross-encoder` (in `haiku.rag`) | |
| `zeroentropy` | | | yes | `zeroentropy` (in `haiku.rag`) | `ZEROENTROPY_API_KEY` |
| `jina` | | | yes | none | `JINA_API_KEY` |
| `jina-local` | | | yes | `jina` (in `haiku.rag`) | |

Any other provider Pydantic AI supports works for QA, passed to it by name. [Installation](../installation.md#extras) lists the extras.

`haiku-rag` and `haiku-ingester` load a `.env` file from the working directory on start, which is a place for these keys. Library use does not read `.env`.

Ollama is reached at `providers.ollama.base_url`, which defaults to `OLLAMA_BASE_URL` or `http://localhost:11434`. A model's own `base_url` overrides it:

```yaml
providers:
  ollama:
    base_url: http://localhost:11434
```

## Model settings

These settings apply to the QA and title models. The picture-description model reads them only under `rebuild --descriptions`, see [Picture handling](processing.md#picture-handling):

```yaml
qa:
  model:
    provider: ollama
    name: qwen3.8
    temperature: 0.3
    max_tokens: 500
```

- **temperature**: Sampling temperature. Unset by default. The built-in QA and title models set 0.3 when their section is omitted.
- **max_tokens**: Maximum tokens in a response. Unset by default, except title generation (100).
- **thinking**: Reasoning on or off, or an effort level, see [Thinking](#thinking).
- **base_url**: Endpoint of an OpenAI-compatible server (vLLM, LM Studio, sglang).
- **api_key**: Key for this endpoint, see [Per-endpoint API keys](#per-endpoint-api-keys).
- **extra_body**: Raw request fields, see [Raw provider pass-through](#raw-provider-pass-through).
- **vision**: Whether the model reads images. Default `false`, and `true` for the built-in QA model when `qa` is omitted.

haiku.rag applies them on the providers it builds directly: `ollama`, `vllm`, `openai`, `openrouter`, `anthropic`, `google`, `groq`, `bedrock` and `mistral`. Any other provider is passed to Pydantic AI by name, which applies none of them, and haiku.rag logs a warning when they are set.

### Per-endpoint API keys

The `openai` provider reads `OPENAI_API_KEY`, so several OpenAI-compatible endpoints would share one key. `api_key` gives each its own, and [variable expansion](index.md#environment-variables) keeps the secret in the environment:

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

`api_key` is honored on the `openai`, `ollama`, `openrouter` and `vllm` providers, on `vllm` and `openrouter` embedders and rerankers, and on the picture-description endpoint, which falls back to `OPENAI_API_KEY` only for the public OpenAI endpoint. Other providers read their own environment variable, and setting `api_key` on them raises.

### Thinking

`thinking` takes `false`, `true`, or an effort level: `minimal`, `low`, `medium`, `high`, `xhigh`. Unset leaves the model's default. `enable_thinking` is its former name, still read with a `FutureWarning` until 0.90.0.

How the value reaches the model depends on the provider:

- **Vendor APIs** (`openai` without a `base_url`, `anthropic`, `google`, `groq`, `bedrock`, `mistral`): passed as Pydantic AI's unified `thinking` setting, which it maps per provider and clamps to what each model offers. See the [Pydantic AI thinking documentation](https://ai.pydantic.dev/thinking/).
- **Self-hosted endpoints** (`ollama`, `vllm`, `openai` with a `base_url`, and the picture-description model): sent as the request's `reasoning_effort` field, whatever the model name. `false` sends `none`, `true` sends `medium`, and a level is sent as written. The server decides what it means, and a chat template may reject a level it does not know. Ollama never rejects one, mapping `minimal` to `low` and `xhigh` to `max`.

Write the level the model accepts:

```yaml
qa:
  model:
    provider: vllm
    name: Inferact/Qwen3.8-27B-NVFP4
    base_url: http://localhost:11439
    thinking: xhigh
```

A chat template with a switch of its own ignores `reasoning_effort` and takes the switch through `extra_body.chat_template_kwargs`. Leave `thinking` unset when you set the switch, since a `chat_template_kwargs` value takes precedence:

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

!!! note "Anthropic thinking and max_tokens"
    Anthropic requires `max_tokens` to exceed the thinking budget, and `thinking: true` requests Pydantic AI's default budget of 10000 tokens. Set `max_tokens` above 10000 on Claude models that use budget-based thinking, or leave it unset on Sonnet 4.6+ and Opus 4.6+, which use adaptive thinking.

`provider: bedrock` uses Bedrock Converse, which does not serve the proprietary OpenAI models and raises for one. `provider: bedrock-mantle` reaches them, passed to Pydantic AI by name, so its settings are not applied.

### Raw provider pass-through

`extra_body` is forwarded verbatim as Pydantic AI's `ModelSettings.extra_body`, and its keys become top-level request fields:

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

It is honored by openai, ollama, openrouter, anthropic, groq and vllm, and ignored by google, bedrock and mistral. Ollama honors only `top_p` of the three above. `extra_body.reasoning_effort` overrides the value `thinking` sends.

## Embedding providers

An embedding model needs `provider`, `name` and `vector_dim`, which must be the model's own output dimension. `embeddings.batch_size` (default `512`) is how many text chunks go in one embedding call during ingest. Lower it for a provider that caps tokens per request. Pictures are embedded one per call.

```yaml
embeddings:
  model:
    provider: ollama            # the default
    name: qwen3-embedding:4b
    vector_dim: 2560
```

```yaml
embeddings:
  model:
    provider: openai
    name: text-embedding-3-small
    vector_dim: 1536
```

```yaml
embeddings:
  model:
    provider: voyageai
    name: voyage-3.5
    vector_dim: 1024
```

```yaml
embeddings:
  model:
    provider: cohere
    name: embed-v4.0
    vector_dim: 1024
```

```yaml
embeddings:
  model:
    provider: sentence-transformers   # in-process, HuggingFace models
    name: all-MiniLM-L6-v2
    vector_dim: 384
```

An OpenAI-compatible server takes `provider: openai` with a `base_url` that includes `/v1`:

```yaml
embeddings:
  model:
    provider: openai
    name: text-embedding-qwen3-embedding-4b
    vector_dim: 2560
    base_url: http://localhost:1234/v1   # LM Studio
```

### Multimodal embedders

`embeddings.model.multimodal: true` puts pictures in the same vector space as text, for text-to-figure and image-as-query search. It is supported on `vllm`, `openrouter`, `voyageai` and `cohere`, and raises on any other provider. The same providers without the flag embed text only.

Only a multimodal embedder produces picture chunks at ingest. Without one, an image-only document produces no chunks and cannot be retrieved. Switching `multimodal` does not change the stored embedding identity and raises no drift error. Re-ingest or rebuild to add or drop picture chunks.

```yaml
embeddings:
  model:
    provider: vllm
    name: Qwen/Qwen3-VL-Embedding-8B
    vector_dim: 4096
    base_url: http://localhost:8000/v1
    multimodal: true
```

`provider: vllm` sends text in the standard `input` field and images in vLLM's `messages` form. Tested with `Qwen/Qwen3-VL-Embedding-8B` (4096) and `jinaai/jina-embeddings-v4` (2048).

```yaml
embeddings:
  model:
    provider: openrouter
    name: nvidia/llama-nemotron-embed-vl-1b-v2:free
    vector_dim: 2048
    multimodal: true
```

`provider: openrouter` defaults `base_url` to `https://openrouter.ai/api/v1`. Tested with `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048), `voyageai/voyage-multimodal-3.5` (1024) and `google/gemini-embedding-2` (3072). Its embedding models are listed at `/v1/embeddings/models`, not `/v1/models`.

VoyageAI's `voyage-multimodal-3` (1024) and Cohere's `embed-v4.0` take the flag the same way.

A data URI passed as a plain string is embedded as text by these endpoints, so an image reaches the image path only through a multimodal embedder. See [Picture handling](processing.md#picture-handling) for how pictures, embedder and QA model combine.

## Question-answering providers

Any provider Pydantic AI supports can answer. Ollama is the default:

```yaml
qa:
  model:
    provider: ollama
    name: qwen3.8
```

```yaml
qa:
  model:
    provider: openai
    name: gpt-4o-mini
```

```yaml
qa:
  model:
    provider: anthropic
    name: claude-haiku-4-5
```

```yaml
qa:
  model:
    provider: openrouter        # one endpoint in front of many vendors
    name: openai/gpt-4o-mini
```

OpenRouter sends `thinking` as its `reasoning` field, and the model decides what it honors.

`provider: vllm` serves a vLLM chat model, with `base_url` written with or without `/v1`. Its model profile, inferred from the model name, merges leading system messages and picks tool-schema behavior. `thinking` does not depend on that inference.

```yaml
qa:
  model:
    provider: vllm
    name: Qwen/Qwen3-4B
    base_url: http://localhost:8002
```

`provider: vllm` under `embeddings.model` and `reranking.model` is a different implementation, haiku.rag's own client for vLLM's embedding and rerank endpoints.

Other OpenAI-compatible servers take `provider: openai` with a `base_url` including `/v1`, and must serve a model that supports tool calling:

```yaml
qa:
  model:
    provider: openai
    name: gpt-oss-20b
    base_url: http://localhost:1234/v1
```

Google, Groq and Mistral follow the same shape, for example `provider: google` with `name: gemini-2.5-flash`. See the [Pydantic AI model list](https://ai.pydantic.dev/models/).

## Reranking providers

A reranker re-orders search results with a dedicated model. There is none by default. Configure `reranking.model` to enable one, and remove it or set `model: null` to disable it again. See [Search settings](qa.md#search-settings) for how it changes what search fetches.

```yaml
reranking:
  model:
    provider: cross-encoder     # any HuggingFace cross-encoder, in-process
    name: Qwen/Qwen3-Reranker-0.6B
```

`cross-encoder` runs any `sentence_transformers.CrossEncoder` model. Also tested: `BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`.

```yaml
reranking:
  model:
    provider: cohere
    name: rerank-v3.5
```

```yaml
reranking:
  model:
    provider: zeroentropy
    name: zerank-1
```

```yaml
reranking:
  model:
    provider: jina              # Jina's HTTP API
    name: jina-reranker-v3
```

```yaml
reranking:
  model:
    provider: jina-local        # in-process
    name: jinaai/jina-reranker-v3
```

The local Jina Reranker v3 model is licensed CC BY-NC 4.0, which restricts commercial use. The API mode has no such restriction.

```yaml
reranking:
  model:
    provider: vllm
    name: Qwen/Qwen3-Reranker-4B
    base_url: http://localhost:8001/v1
```

`provider: vllm` posts to vLLM's `/v1/rerank`, with `base_url` written with or without `/v1`.

### Multimodal reranking

`reranking.multimodal: true` scores picture chunks by their image bytes as well as their text. It is supported on `vllm` and `openrouter`, and the model must accept images:

```yaml
reranking:
  multimodal: true
  model:
    provider: vllm
    name: nvidia/llama-nemotron-rerank-vl-1b-v2
    base_url: http://localhost:8001/v1
```

Picture chunks go as image documents beside the text documents in one rerank request. A chunk with neither text nor picture bytes is not sent.

On OpenRouter, `base_url` defaults to `https://openrouter.ai/api/v1`, and `nvidia/llama-nemotron-rerank-vl-1b-v2:free` is the reranker that takes images. Its text-only rerankers reject a document with no text. Rerank models are listed at `/v1/models?output_modalities=rerank`.
