# Search and Question Answering

## Search Settings

Configure search behavior and context expansion:

```yaml
search:
  limit: 5                     # Default number of results to return
  context_radius: 0            # DocItems before/after to include for text content
  max_context_items: 10        # Maximum items in expanded context
  max_context_chars: 10000     # Maximum characters in expanded context
```

- **limit**: Default number of search results to return when no limit is specified. Used by CLI, MCP server, QA, and research workflows. Default: 5
- **context_radius**: For text content (paragraphs), includes N DocItems before and after. Set to 0 to disable expansion (default).
- **max_context_items**: Limits how many document items (paragraphs, list items, etc.) can be included in expanded context. Default: 10.
- **max_context_chars**: Hard limit on total characters in expanded content. Default: 10000.

Structural content (tables, code blocks, lists) uses type-aware expansion that automatically includes the complete structure regardless of how it was chunked.

!!! note "Reranking behavior"
    When a reranker is configured, search automatically retrieves 10x the requested limit, then reranks to return the final count. This improves result quality without requiring you to adjust `limit`.

## Question Answering Configuration

Configure the QA workflow:

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: false
  max_sub_questions: 3  # Maximum sub-questions for deep QA
  max_iterations: 2     # Maximum search iterations per sub-question
  max_concurrency: 1    # Sub-questions processed in parallel
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **max_sub_questions**: For deep QA mode, maximum number of sub-questions to generate (default: 3)
- **max_iterations**: Maximum search/evaluate cycles per sub-question (default: 2)
- **max_concurrency**: Number of sub-questions to process in parallel (default: 1)

Deep QA mode (`haiku-rag ask --deep`) decomposes complex questions into sub-questions, processes them in parallel batches, and synthesizes the results.

## Research Configuration

Configure the multi-agent research workflow:

```yaml
research:
  model:
    provider: ""            # Empty to use qa settings
    name: ""               # Empty to use qa model
    enable_thinking: false
  max_iterations: 3
  confidence_threshold: 0.8
  max_concurrency: 1
```

- **model**: LLM configuration. Leave provider/model empty to inherit from `qa` (see [Providers](providers.md#model-settings))
- **max_iterations**: Maximum search/evaluate cycles (default: 3)
- **confidence_threshold**: Stop when confidence score meets/exceeds this (default: 0.8)
- **max_concurrency**: Sub-questions searched in parallel per iteration (default: 1)

The research workflow plans sub-questions, searches in parallel batches, evaluates findings, and iterates until reaching the confidence threshold or max iterations.
