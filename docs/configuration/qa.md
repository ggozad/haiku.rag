# Search and question answering

## Search settings

Configure search behavior and context expansion:

```yaml
search:
  limit: 5                     # Default number of results to return
  max_context_chars: 5000     # Maximum characters in expanded context
  rerank_pool_factor: 10      # Candidates per result a reranker scores
  fts_language: English       # Stemmer and normalization of the full-text index
```

- **limit**: Default number of search results to return when no limit is specified. Used by CLI, MCP server, and QA. Default: 5
- **max_context_chars**: Hard limit on total characters in expanded content. Default: 5000.
- **rerank_pool_factor**: With a reranker configured, a text search fetches `limit × rerank_pool_factor` candidates and reranks them down to `limit`. Default: 10.
- **fts_language**: The language of the full-text index: its stemmer, and how it lowercases and folds accents. One of Arabic, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish. It applies when the index is built: on a new database, and on `haiku-rag rebuild` in any mode but `--title-only` and `--set-embedder`. A write rebuilds the index only when it covers no rows, so an existing database takes a new language on `haiku-rag rebuild --reindex`. Queries are tokenized by the index, so they always match it. One database has one language. Default: English.

Context expansion is section-aware. The capability's search tool, the MCP search tools and in-code `search()` expand every result. `HaikuRAG.search` returns chunks as matched, and `HaikuRAG.expand_context` expands them. For structured documents (with section headers), expansion includes the entire section containing the match when it fits the budget. A section larger than the budget grows outward item-by-item from the match, staying inside the section. A section under 20% of the budget (e.g., a title+authors area) grows across section boundaries until the budget is filled, except when the match is a picture or table, which returns its section as-is. Noise labels (page headers, page footers, table of contents) are skipped. For unstructured documents, expansion grows outward item-by-item. Results without `doc_item_refs` (e.g., custom chunks passed to `import_document`) pass through unexpanded. Results whose expanded ranges overlap within a document are merged into one.

!!! note "Reranking behavior"
    With a reranker configured, a text search retrieves `rerank_pool_factor` times the requested limit and reranks down to it. Image queries skip the reranker.

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
  max_images: null      # Maximum images in one model request (null: no limit)
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **model.vision**: Set to `true` for vision-capable models (`qwen2.5vl`, `qwen3.6`, `gpt-4o`, `claude-sonnet`, …). The capability's `search` tool only attaches picture bytes (`BinaryContent`) to its `ToolReturn` when this is `true`, otherwise picture bytes are withheld. See [Pictures × embedder × QA model](processing.md#pictures-embedder-qa-model-how-the-pieces-compose) for the full matrix.
- **max_searches**: Maximum number of search units a capability can spend per question (default: 5). Up to three searches emitted in the same model response share one unit, so a model that rephrases its query in one response spends one unit. A search in a later response starts a new unit, as does each further group of three within one response. Searches in one response also deduplicate their returns: evidence a sibling search already showed collapses to a reference line, and each picture attaches once per response.
- **max_executions**: Maximum `execute_code` calls per question before the capability is told to answer from what it has (default: 15)
- **max_images**: Maximum number of images in one model request (default: unset, no limit). Set it at or below the limit of the model server, for example vLLM's `--limit-mm-per-prompt`. See [Image limit](#image-limit).

!!! note "Thinking on self-hosted models"
    `thinking` is sent as `reasoning_effort` to every self-hosted endpoint, and the accepted levels are the model's own. A chat template with a switch of its own takes it through `extra_body`. See [Thinking](providers.md#thinking).

## Image limit

A model server refuses a request holding more images than it allows. The pictures of every search stay in the conversation, and [evidence compaction](../capabilities/compaction.md) re-attaches the pictures cited in earlier questions, so a long question or a long conversation can exceed the server's limit.

With `qa.max_images` set, the capability drops retrieved pictures from a request that holds more images than the limit:

- Every image in the request counts: retrieved pictures, images attached by the user, and images returned by other tools.
- Only retrieved pictures are dropped, each with its label. The text of the search result or of the cited evidence stays.
- Pictures from earlier rounds of searches are dropped first, starting with the oldest round. Pictures cited in earlier questions are dropped next. The pictures of the newest round of searches are dropped last.
- Within one search, the picture of the lowest-ranked result is dropped first. Among cited pictures, the picture of the oldest citation is dropped first.
- The stored conversation keeps every picture. The limit applies to the request only.

Images attached by the user and images from other tools are never dropped. A request with more of those images than the limit still exceeds it. Images added by the `prepare_model_request` hook of a capability registered after the RAG capability are not counted.

## Sandbox configuration

Limits of one `execute_code` call, in the RAG capability and in the MCP `execute_code` tool:

```yaml
sandbox:
  code_timeout: 60.0      # Per call: compute stops, no read or search starts past it
  max_output_chars: 50000 # Truncate output after this many chars
```

- **code_timeout**: Seconds a single `execute_code` call has (default: 60). Past it the sandbox starts no further host call, a document read or an in-code `search()` / `list_documents()`; one already running finishes. Computation past the limit is stopped by the interpreter, which resets the session: variables from earlier calls are gone.
- **max_output_chars**: Truncate code output after this many characters (default: 50000)

See [RAG capability](../capabilities/rag.md) for usage details.
