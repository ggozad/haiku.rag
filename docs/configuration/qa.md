# Search and Question Answering

## Search Settings

Configure search behavior and context expansion:

```yaml
search:
  limit: 5                     # Default number of results to return
  max_context_chars: 5000     # Maximum characters in expanded context
```

- **limit**: Default number of search results to return when no limit is specified. Used by CLI, MCP server, and QA. Default: 5
- **max_context_chars**: Hard limit on total characters in expanded content. Default: 5000.

Context expansion is automatic and section-aware. For structured documents (with section headers), expansion includes the entire section containing the match. For sections that exceed the budget or are too small (e.g., a title+authors area), expansion grows outward item-by-item from the match center, skipping noise labels (footnotes, page headers). This naturally crosses into adjacent sections until the budget is filled. Picture and table matches are exempt: they return their enclosing section as-is and never cross section boundaries. For unstructured documents, expansion grows outward item-by-item. Results without `doc_item_refs` (e.g., custom chunks passed to `import_document`) pass through unexpanded.

!!! note "Reranking behavior"
    When a reranker is configured, search automatically retrieves 10x the requested limit, then reranks to return the final count. This improves result quality without requiring you to adjust `limit`.

## Question Answering Configuration

Configure the RAG capability (used by `client.ask`, `haiku-rag ask`, and the MCP `ask_question` tool):

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: true
    temperature: 0.3          # Default: 0.3
    vision: false             # Set true for vision-capable models
  max_searches: 5       # Maximum search tool calls per question
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **model.vision**: Set to `true` for vision-capable models (`qwen2.5vl`, `qwen3.6`, `gpt-4o`, `claude-sonnet`, …). The capability's `search` tool only attaches picture bytes (`BinaryContent`) to its `ToolReturn` when this is `true`, otherwise picture bytes are withheld. See [Pictures × embedder × QA model](processing.md#pictures-embedder-qa-model-how-the-pieces-compose) for the full matrix.
- **max_searches**: Maximum number of search tool calls a capability can make per question (default: 5). Shared by the RAG and analysis capabilities.

!!! note "Thinking on vLLM"
    `enable_thinking` only applies to models with a pydantic-ai reasoning profile (o-series, gpt-5, gpt-oss). For other vLLM-served models such as Qwen3 or the Gemma family, the field is a silent no-op — set the chat template switch via [`extra_body`](providers.md#raw-provider-pass-through) instead.

## Analysis Configuration

Configure the analysis capability:

```yaml
analysis:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
    temperature: 0.0        # Default: 0.0 (deterministic for code generation)
  code_timeout: 60.0      # Max seconds a call may spend reading documents
  max_output_chars: 50000 # Truncate output after this many chars
  max_executions: 15      # Max execute_code calls per question
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings)). When unset, falls back to `qa.model`.
- **code_timeout**: Seconds a single `execute_code` call may spend reading documents (default: 60). The sandbox refuses further reads past this point. Code that computes without reading is killed by the worker watchdog at the same limit. `code_timeout * max_executions` is the cumulative ceiling across all calls in one question.
- **max_output_chars**: Truncate code output after this many characters (default: 50000)
- **max_executions**: Maximum `execute_code` calls per question before the capability is told to answer from what it has (default: 15)

See [Analysis capability](../capabilities/analysis.md) for usage details.
