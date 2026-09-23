# Search and question answering

## Search settings

Configure search behavior and context expansion:

```yaml
search:
  limit: 5                     # Default number of results to return
  max_context_chars: 5000     # Maximum characters in expanded context
```

- **limit**: Default number of search results to return when no limit is specified. Used by CLI, MCP server, and QA. Default: 5
- **max_context_chars**: Hard limit on total characters in expanded content. Default: 5000.

Context expansion is section-aware. The capability's search tool, the MCP search tools and in-code `search()` expand every result. `HaikuRAG.search` returns chunks as matched, and `HaikuRAG.expand_context` expands them. For structured documents (with section headers), expansion includes the entire section containing the match when it fits the budget. A section larger than the budget grows outward item-by-item from the match, staying inside the section. A section under 20% of the budget (e.g., a title+authors area) grows across section boundaries until the budget is filled, except when the match is a picture or table, which returns its section as-is. Noise labels (page headers, page footers, table of contents) are skipped. For unstructured documents, expansion grows outward item-by-item. Results without `doc_item_refs` (e.g., custom chunks passed to `import_document`) pass through unexpanded. Results whose expanded ranges overlap within a document are merged into one.

!!! note "Reranking behavior"
    With a reranker configured, a text search retrieves 10x the requested limit and reranks down to it. Image queries skip the reranker.

## Question answering configuration

Configure the RAG capability (used by `client.ask` and `haiku-rag ask`):

```yaml
qa:
  model:
    provider: ollama
    name: qwen3.8
    thinking: true
    temperature: 0.3          # Default: 0.3
    vision: true              # Set false for text-only models
  max_searches: 5       # Maximum search units per question
  max_executions: 15    # Maximum execute_code calls per question
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **model.vision**: Set to `true` for vision-capable models (`qwen2.5vl`, `qwen3.6`, `gpt-4o`, `claude-sonnet`, …). The capability's `search` tool only attaches picture bytes (`BinaryContent`) to its `ToolReturn` when this is `true`, otherwise picture bytes are withheld. See [Pictures × embedder × QA model](processing.md#pictures-embedder-qa-model-how-the-pieces-compose) for the full matrix.
- **max_searches**: Maximum number of search units a capability can spend per question (default: 5). Up to three searches emitted in the same model response share one unit, so a model that rephrases its query in one response spends one unit. A search in a later response starts a new unit, as does each further group of three within one response. Searches in one response also deduplicate their returns: evidence a sibling search already showed collapses to a reference line, and each picture attaches once per response.
- **max_executions**: Maximum `execute_code` calls per question before the capability is told to answer from what it has (default: 15)

!!! note "Thinking on self-hosted models"
    `thinking` is sent as `reasoning_effort` to every self-hosted endpoint, and the accepted levels are the model's own. A chat template with a switch of its own takes it through `extra_body`. See [Thinking](providers.md#thinking).

## Sandbox configuration

Limits of one `execute_code` call, in the RAG capability and in the MCP `execute_code` tool:

```yaml
sandbox:
  code_timeout: 60.0      # Per call: compute stops, no read or search starts past it
  max_output_chars: 50000 # Truncate output after this many chars
```

- **code_timeout**: Seconds a single `execute_code` call has (default: 60). Past it the sandbox starts no further host call, a document read or an in-code `search()` / `list_documents()`; one already running finishes. Code that computes without host calls is killed by the worker watchdog at the same limit. `code_timeout * qa.max_executions` is the cumulative ceiling across all calls in one question.
- **max_output_chars**: Truncate code output after this many characters (default: 50000)

See [RAG capability](../capabilities/rag.md) for usage details.
