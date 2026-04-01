# Tuning

How to adjust haiku.rag's pipeline for better retrieval and answer quality. For individual setting definitions and defaults, see [Configuration](configuration/index.md).

## Pipeline Overview

Documents flow through: **chunking → embedding → hybrid search (vector + FTS) → reranking → context expansion → LLM generation**. Retrieval tuning (chunking through reranking) is highest-leverage — if the LLM never sees the right chunks, no prompt or model change will help.

## Tuning Retrieval

### Chunking

`chunk_size` controls the granularity of retrieval. Smaller chunks match queries more precisely but carry less context each; larger chunks provide more surrounding information but dilute relevance signals. On the Wix benchmark, increasing from 256 to 512 tokens raised MAP from 0.43 to 0.45 on plain text — a modest gain that also increases token cost per result. See [Processing](configuration/processing.md#chunk-size) for configuration.

`chunker_type` selects between `hybrid` (default) and `hierarchical` chunking. Hierarchical chunking preserves the document's heading structure and works better for deeply nested or structured content. See [Chunking Strategies](configuration/processing.md#chunking-strategies).

### Embedding Model

Larger embedding models produce better representations at the cost of slower indexing and more storage. The choice of embedding model has a larger impact on retrieval quality than most other settings. See [Providers](configuration/providers.md) for available options and [Benchmarks](benchmarks.md) for real comparisons across models.

### Reranking

When configured, a cross-encoder reranker re-scores 10x the requested candidates and returns the top results. This adds latency but improves precision — on the Wix benchmark, adding `mxbai-rerank-base-v2` raised MAP from 0.34 to 0.39 on HTML content. See [Search Settings](configuration/qa-research.md#search-settings) for how reranking integrates with search.

### Search Settings

`limit` controls how many results reach the LLM. More candidates improve recall but increase token usage. See [Search Settings](configuration/qa-research.md#search-settings).

`context_radius` expands text chunks with neighboring document items. Structural content (tables, code blocks, lists) expands automatically to include the complete structure. This setting matters most with small `chunk_size` values, where individual chunks may lack sufficient context. `max_context_items` and `max_context_chars` cap expansion to prevent context bloat.

## Tuning Generation

Model and temperature selection affect answer quality directly — see [Providers](configuration/providers.md#model-settings) for options.

`domain_preamble` prepends domain context to all agent prompts — including the main agent, skill subagents, and internal agents (QA, research). Use it to describe what the knowledge base contains and clarify domain-specific terminology. For full prompt replacement, set `prompts.qa` directly. See [Prompt Customization](configuration/prompts.md).

For automated prompt optimization, see [Prompt Optimization (GEPA)](#prompt-optimization-gepa) below.

## What Requires a Rebuild

| Change | Rebuild required? |
|--------|:-:|
| `chunk_size`, `chunker_type`, `chunking_merge_peers` | Yes — `haiku-rag rebuild` |
| Embedding model | Yes — `haiku-rag rebuild` |
| Search settings, reranking, prompts | No |

## Measuring Changes

Use the inspector for ad-hoc exploration:

```bash
haiku-rag inspect
```

For systematic measurement, use the `evaluations/` workspace which provides retrieval metrics (MRR, MAP) and LLM-judged QA accuracy via `pydantic-evals`:

```bash
# Run retrieval + QA benchmarks
evaluations run <dataset>

# Skip database rebuild when only changing search/reranking/prompt settings
evaluations run <dataset> --skip-db

# Limit test cases for faster iteration
evaluations run <dataset> --limit 50
```

See [Benchmarks](benchmarks.md) for dataset details, methodology, and baseline results.

## Prompt Optimization (GEPA)

The `evaluations optimize` command uses GEPA (Generalized Evolutionary Prompt Algorithm) to evolve the QA system prompt. It evaluates candidates on minibatches scored by an LLM judge, reflects on failures, proposes mutations, and accepts improvements.

```bash
# Basic optimization
evaluations optimize wix

# Constrained run
evaluations optimize repliqa --limit 40 --num-candidates 30

# Save result
evaluations optimize wix --output optimized_prompt.txt
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | all cases | QA cases to use (split 50/50 train/val) |
| `--num-candidates` | `50` | Number of candidate prompts to evaluate |
| `--output` | — | Save optimized prompt to file |
| `--config` | auto | haiku.rag YAML config path |
| `--db` | auto | Database path override |
| `--judge-model` | `config.qa.model` | LLM judge as `provider:name` |
| `--reflect-model` | `config.qa.model` | Reflection LLM as `provider:name` |

Apply the result in your config:

```yaml
prompts:
  qa: |
    Your optimized prompt text here...
```

Or programmatically: `get_qa_agent(client, config, system_prompt=optimized_prompt)`.
