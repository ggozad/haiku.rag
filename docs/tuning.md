# Tuning haiku.rag for Your Corpus

This guide explains how to tune haiku.rag settings based on your document corpus characteristics. The right settings depend on your document types, query patterns, and accuracy requirements.

## Key Concepts

### Retrieval vs Generation

RAG has two phases:

1. **Retrieval**: Finding relevant chunks from your corpus
2. **Generation**: Using those chunks to answer questions

Poor retrieval means the LLM never sees the relevant content, regardless of how good the model is. Tuning retrieval is usually more impactful than tuning generation.

### Recall vs Precision

- **Recall**: What fraction of relevant documents did we find?
- **Precision**: What fraction of retrieved documents are relevant?

For RAG, recall matters more than precision. Missing a relevant chunk means wrong answers. Including an extra irrelevant chunk just wastes context tokens.

## Search Settings

### `search.limit`

Default number of chunks to retrieve.

```yaml
search:
  limit: 5  # Default
```

**When to increase:**

- Complex questions requiring information from multiple sources
- Broad topics spread across many documents

**When to decrease:**

- Simple factual questions
- Highly focused corpus where top results are usually correct
- Cost-sensitive deployments (fewer chunks = fewer tokens)

**Typical values:** 3-10

### `search.context_radius`

Number of adjacent DocItems to include when expanding search results. Only applies to text content (paragraphs). Tables, code blocks, and lists use structural expansion automatically.

```yaml
search:
  context_radius: 0  # Default: no expansion
```

**When to increase:**

- Answers require surrounding context (definitions, explanations)
- Chunks are small and queries need more context
- Documents have strong local coherence (adjacent paragraphs relate)

**When to keep at 0:**

- Large chunks that already contain sufficient context
- Documents where adjacent content is often unrelated
- When chunk boundaries align well with semantic units

**Typical values:** 0-3

### `search.max_context_items` and `search.max_context_chars`

Safety limits on context expansion to prevent runaway expansion.

```yaml
search:
  max_context_items: 10     # Max DocItems per expanded result
  max_context_chars: 10000  # Max characters per expanded result
```

Increase if expansion is being truncated and you need more context. Decrease if expanded results are too long for your LLM context window.

## Processing Settings

### `processing.chunk_size`

Maximum tokens per chunk (using the configured tokenizer).

```yaml
processing:
  chunk_size: 256  # Default
```

**Trade-offs:**

| Smaller chunks (128-256) | Larger chunks (512-1024) |
|-------------------------|-------------------------|
| More precise retrieval | Better context per chunk |
| May miss spanning content | Better recall |
| More chunks to search | Faster search |
| Better for specific queries | Better for broad queries |

**Guidance by corpus type:**

- **Technical documentation**: 256-512 (specific lookups)
- **Long-form articles**: 512-1024 (need context)
- **FAQs/short answers**: 128-256 (discrete answers)
- **Code documentation**: 256-512 (function-level)

### `processing.chunker_type`

Chunking strategy.

```yaml
processing:
  chunker_type: hybrid  # Default
```

- **`hybrid`**: Structure-aware with token limits. Best for most documents.
- **`hierarchical`**: Preserves document hierarchy strictly. Use for highly structured documents where hierarchy matters.


### `processing.chunking_merge_peers`

Whether to merge adjacent small chunks that share the same section.

```yaml
processing:
  chunking_merge_peers: true  # Default
```

Keep `true` unless you specifically want very granular chunks. Merging improves embedding quality by ensuring chunks have sufficient context.

## Embedding Settings

### Model Selection

Embedding model choice significantly impacts retrieval quality.

```yaml
embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560
```

**Considerations:**

- Larger models generally produce better embeddings but are slower
- Match `vector_dim` to your model's actual output dimension
- Local models (Ollama) vs API models (OpenAI, VoyageAI) trade-off cost vs quality

### Contextualizing Embeddings

Chunks are embedded with section headings prepended (via `contextualize()`). This improves retrieval by including structural context in the embedding.

If your documents lack clear headings, embeddings will be based on chunk content alone.

## Reranking

Reranking retrieves more candidates than needed, then uses a cross-encoder to re-score them.

```yaml
reranking:
  model:
    provider: mxbai  # or cohere, zeroentropy, vllm
    name: mixedbread-ai/mxbai-rerank-base-v2
```

**When to use reranking:**

- Embedding model has limited accuracy
- Queries are complex or ambiguous
- You can afford the latency (adds ~100-500ms)

**When to skip reranking:**

- Simple, specific queries
- High-quality embedding model
- Latency-sensitive applications

When reranking is enabled, haiku.rag automatically retrieves 10x the requested limit, then reranks to the final count. You don't need to adjust `search.limit` for reranking.

## Tuning Workflow

### 1. Use the Inspector

The inspector is your best tool for understanding how your corpus is chunked and how search behaves:

```bash
haiku-rag inspect
```

**What to look for:**

- Browse documents and their chunks to see how content is split
- Use the search modal (`/`) to test queries and see which chunks are retrieved
- Press `c` on a chunk to view expanded context - see what additional content would be included with `context_radius > 0`
- Check chunk sizes - are they too small (fragmented) or too large (unfocused)?

### 2. Test Search Manually

Before changing settings, run searches from the CLI to understand current behavior:

```bash
# Search and see results
haiku-rag search "your test query" --limit 10

# Try the QA to see end-to-end behavior
haiku-rag ask "your question"
```

### 3. Identify the Bottleneck

- **Relevant chunks not retrieved**: Try larger `search.limit`, smaller `chunk_size`, or a different embedding model
- **Too many irrelevant chunks**: Try reranking or larger `chunk_size`
- **Chunks found but answers wrong**: Try `context_radius` expansion or a better QA model

### 4. Test One Change at a Time

```bash
# After changing chunk_size, rebuild is required
haiku-rag rebuild

# After changing search settings, no rebuild needed - just test again
haiku-rag search "your test query"
```

### 5. Build Dataset-Specific Evaluations

For systematic tuning, create evaluations specific to your corpus. See the `evaluations/` directory in the repository for examples of how to:

- Define test cases with questions and expected answers
- Run retrieval benchmarks (MRR, MAP)
- Run QA accuracy benchmarks with LLM judges

Custom evaluations let you measure the impact of configuration changes objectively rather than relying on intuition.

### 6. Consider Your Corpus

| Corpus Type | Suggested Starting Point |
|-------------|-------------------------|
| Technical docs | `chunk_size: 256`, `limit: 10`, `context_radius: 1` |
| Legal/contracts | `chunk_size: 512`, `limit: 5`, `context_radius: 2` |
| News articles | `chunk_size: 512`, `limit: 5`, `context_radius: 0` |
| Scientific papers | `chunk_size: 256`, `limit: 5`, reranking enabled |
| FAQs | `chunk_size: 128`, `limit: 5`, `context_radius: 0` |
| Code repos | `chunk_size: 256`, `limit: 10`, `context_radius: 1` |

## Common Issues

### "Relevant content not being retrieved"

1. Check chunk boundaries - is the content split awkwardly?
2. Try smaller chunks for more granular matching
3. Increase `search.limit`
4. Consider a different embedding model

### "Retrieved chunks lack context"

1. Increase `context_radius` for text content
2. Increase `chunk_size` for more context per chunk
3. Structural content (tables, code) expands automatically

### "Search is slow"

1. Create a vector index: `haiku-rag create-index`
2. Reduce `search.limit`
3. Consider a smaller embedding model

### "QA answers are wrong despite good retrieval"

1. Check if chunks are being truncated by LLM context limits
2. Try a more capable QA model
3. Reduce number of chunks or expansion to fit context window

## Example Configurations

### High-Precision Technical Documentation

```yaml
processing:
  chunk_size: 256
  chunker_type: hybrid

search:
  limit: 10
  context_radius: 1
  max_context_items: 15

reranking:
  model:
    provider: mxbai
    name: mixedbread-ai/mxbai-rerank-base-v2
```

### Long-Form Content (Articles, Reports)

```yaml
processing:
  chunk_size: 512
  chunker_type: hybrid

search:
  limit: 5
  context_radius: 2
  max_context_items: 10
```

### FAQ/Knowledge Base

```yaml
processing:
  chunk_size: 128
  chunker_type: hybrid

search:
  limit: 5
  context_radius: 0
```
