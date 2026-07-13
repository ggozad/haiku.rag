# Hooks

Hooks let external packages observe document writes and transform searches without forking haiku.rag. Use them for query rewriting, result annotation, or maintaining state derived from the corpus (a synonym table, an entity index, corpus statistics).

A hook is a class registered under the `haiku.rag.hooks` entry-point group and activated by name in config. Hooks run everywhere the client runs: CLI, MCP server, skills, and your own code.

## Hook points

Subclass `haiku.rag.hooks.Hook` and override any subset:

| Method | Fires | Use for |
|--------|-------|---------|
| `after_ingest(client, document)` | A document's content was written (create, import, batch import, update) | Deriving state from documents |
| `after_delete(client, document_id)` | A document was deleted, once per document in a cascade | Cleaning up derived state |
| `before_search(client, query, filter)` | Before retrieval, text queries only | Query expansion, filter injection |
| `after_search(client, query, results)` | After retrieval, reranking, and deduplication | Annotating, reordering, or filtering results |

`before_search` returns the `(query, filter)` pair to search with. The returned query feeds both the vector and the full-text side. `after_search` returns the result list. Hooks run in the order listed in config, each receiving the previous hook's output.

Hooks receive the `HaikuRAG` client, so they can search, read repositories, and store their own state.

## Registering a hook

```python
from haiku.rag.hooks import Hook

class AbbreviationHook(Hook):
    async def before_search(self, client, query, filter):
        expanded = my_glossary.expand(query)
        return expanded, filter

    async def after_search(self, client, query, results):
        for result in results:
            result.annotations = [
                f"{term}: {definition}"
                for term, definition in my_glossary.definitions_in(result.content)
            ]
        return results
```

Register a zero-arg factory in your package's `pyproject.toml`:

```toml
[project.entry-points."haiku.rag.hooks"]
abbreviations = "my_package.hooks:AbbreviationHook"
```

Activate it in `haiku.rag.yaml`:

```yaml
hooks:
  - abbreviations
```

An unknown name in `hooks:` raises `ValueError` when the client is constructed, so misconfiguration fails at startup. Entry points load lazily. Only the hooks the config references are imported.

## Result annotations

`after_search` hooks can attach free-text notes on `SearchResult.annotations`. Annotations survive context expansion (merged results union the notes of their constituents, deduplicated) and render as `Note: text` lines in the agent-facing output used by the QA skills. MCP responses carry the field as part of the `SearchResult` model. This keeps the context cost proportional to what was retrieved instead of the size of your vocabulary.

## Semantics

- **Update equals ingest.** `after_ingest` fires for both creation and content updates. Treat it as "replace any state you derived from this document". Metadata-only and title-only updates do not fire.
- **Hooks run after the write commits.** They execute outside the store's write lock, so a hook may itself write to the database, and a hook failure never rolls back the document write.
- **Rebuild does not fire hooks.** `rebuild` re-chunks and re-embeds but never changes document content, so content-derived state is unaffected.
- **Backfill is your loop.** A hook enabled on an existing database can backfill by iterating `client.list_documents()` and calling its own `after_ingest`.
- **State lives in the database.** Hooks may create their own LanceDB tables via `client.store`. Prefix table names with `hook_` so they never collide with core tables or future migrations. State then travels with the database and its backups.
