# Search and Question Answering

## Search Settings

Configure search behavior and context expansion:

```yaml
search:
  limit: 10                    # Default number of results to return
  max_context_items: 10        # Maximum items in expanded context
  max_context_chars: 10000     # Maximum characters in expanded context
```

- **limit**: Default number of search results to return when no limit is specified. Used by CLI, MCP server, QA, and research workflows. Default: 10
- **max_context_items**: Limits how many document items (paragraphs, list items, etc.) can be included in expanded context. Default: 10.
- **max_context_chars**: Hard limit on total characters in expanded content. Default: 10000.

Context expansion is automatic and section-aware. For structured documents (with section headers), expansion includes the entire section containing the match. For sections that exceed the budget or are too small (e.g., a title+authors area), expansion grows outward item-by-item from the match center, skipping noise labels (footnotes, page headers) — this naturally crosses into adjacent sections until the budget is filled. For unstructured documents, expansion grows outward item-by-item. Results without `doc_item_refs` (e.g., custom chunks passed to `import_document`) pass through unexpanded.

!!! note "Reranking behavior"
    When a reranker is configured, search automatically retrieves 10x the requested limit, then reranks to return the final count. This improves result quality without requiring you to adjust `limit`.

## Question Answering Configuration

Configure the QA workflow:

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: true
    temperature: 0.3          # Default: 0.3
  max_searches: 3       # Maximum search tool calls per question
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **max_searches**: Maximum number of search tool calls the QA agent can make per question (default: 3)

## Research Configuration

Configure the multi-agent research workflow:

```yaml
research:
  model:
    provider: ""            # Empty to use qa settings
    name: ""               # Empty to use qa model
    enable_thinking: false
    temperature: 0.3        # Default: 0.3
  max_iterations: 3
  max_concurrency: 1
```

- **model**: LLM configuration. Leave provider/model empty to inherit from `qa` (see [Providers](providers.md#model-settings))
- **max_iterations**: Maximum planning/search iterations (default: 3)
- **max_concurrency**: Concurrent search operations (default: 1)

The research workflow uses an iterative feedback loop: the planner proposes one question at a time, sees the answer, then decides whether to continue or synthesize. This continues until the planner marks research as complete or `max_iterations` is reached.

## RLM Configuration

Configure the RLM (Recursive Language Model) agent:

```yaml
rlm:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
    temperature: 0.0        # Default: 0.0 (deterministic for code generation)
  code_timeout: 60.0      # Max seconds for code execution
  max_output_chars: 50000 # Truncate output after this many chars
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **code_timeout**: Maximum seconds for each code execution (default: 60)
- **max_output_chars**: Truncate code output after this many characters (default: 50000)

See [RLM Agent](../agents/rlm.md) for usage details.
