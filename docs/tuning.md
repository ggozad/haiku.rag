# Tuning

How to adjust haiku.rag's pipeline for better retrieval and answer quality. For individual setting definitions and defaults, see [Configuration](configuration/index.md).

For ingester-side tuning (worker count, lease TTL and heartbeat, retry policy, backpressure, circuit breakers), see [Ingester → Workers and retry](ingester.md#workers-and-retry).

## Pipeline overview

Documents flow through: **chunking → embedding → hybrid search (vector + FTS) → reranking → context expansion → LLM generation**. Retrieval (chunking through reranking) is where tuning matters most. If the LLM never sees the right chunks, no prompt or model change will help.

## Tuning retrieval

### Chunking

`chunk_size` controls the granularity of retrieval. Smaller chunks match queries more precisely but carry less context each. Larger chunks provide more surrounding information but dilute relevance signals. See [Processing](configuration/processing.md#chunk-size) for configuration.

`chunker_type` selects between `hybrid` (default) and `hierarchical` chunking. Hierarchical chunking preserves the document's heading structure and works better for deeply nested or structured content. See [Chunking Strategies](configuration/processing.md#chunking-strategies).

### Embedding model

Larger embedding models produce better representations at the cost of slower indexing and more storage. The choice of embedding model has a larger impact on retrieval quality than most other settings. See [Providers](configuration/providers.md) for available options and [Benchmarks](benchmarks.md) for measured comparisons across models.

### Reranking

A reranker adds latency and improves precision. See [Search settings](configuration/qa.md#search-settings) for how it integrates with search, and [Reranking providers](configuration/providers.md#reranking-providers).

### Search settings

`limit` controls how many results reach the LLM. More candidates improve recall but increase token usage. See [Search Settings](configuration/qa.md#search-settings).

Each result is expanded with surrounding content from its section, capped by `max_context_chars`. See [how expansion works](configuration/qa.md#search-settings).

## Tuning generation

Model and temperature selection affect answer quality directly. See [Providers](configuration/providers.md#model-settings) for options.

`domain_preamble` prepends domain context to the RAG capability instructions. Use it to describe what the knowledge base contains and clarify domain-specific terminology. See [Prompt Customization](configuration/prompts.md).

## What requires a rebuild

| Change | What to run |
|--------|-------------|
| `chunk_size`, `chunker_type`, `chunking_tokenizer`, `chunking_merge_peers`, `chunking_use_markdown_tables` | `haiku-rag rebuild --rechunk` |
| Converter, `conversion_options` | `haiku-rag rebuild` |
| `processing.pictures` | See [switching modes](configuration/processing.md#picture-handling) |
| Embedding model or `vector_dim` | `haiku-rag rebuild --embed-only` |
| `search.vector_index_metric` | `haiku-rag create-index`, when an index exists |
| Other search settings, reranking, prompts | Nothing |

## Inspector

`haiku-rag inspect` runs the same search and context expansion the RAG capability uses. Its context-expansion view (`c` on a chunk) shows whether `chunk_size`, `chunker_type` and `max_context_chars` deliver the surrounding content a model needs. See [Inspector](chat.md#inspector).

## Measuring changes

For systematic measurement, use the `evaluations/` workspace which provides retrieval metrics (MAP, Recall, nDCG) and LLM-judged QA accuracy via `pydantic-evals`:

```bash
# Run retrieval + QA benchmarks
evaluations run <dataset>

# Skip database rebuild when only changing search/reranking/prompt settings
evaluations run <dataset> --skip-db

# Limit test cases for faster iteration
evaluations run <dataset> --limit 50
```

See [Benchmarks](benchmarks.md) for dataset details, methodology, and baseline results.
