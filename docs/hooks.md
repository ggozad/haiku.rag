# Hooks

Hooks let external packages observe document writes and transform searches without forking haiku.rag. Use them for query rewriting, result annotation, or maintaining state derived from the corpus (a synonym table, an entity index, corpus statistics).

A hook is a class registered under the `haiku.rag.hooks` entry-point group and activated by name in config. Hooks run everywhere the client runs: CLI, MCP server, skills, and your own code.

## Hook points

Subclass `haiku.rag.hooks.Hook` and override any subset:

| Method | Fires | Use for |
|--------|-------|---------|
| `after_ingest(client, event)` | Document content was written (create, import, batch import, update) | Deriving state from documents |
| `after_delete(client, event)` | Documents were deleted | Cleaning up derived state |
| `before_search(client, request)` | Before retrieval, text queries only | Query expansion, filter injection |
| `after_search(client, request, results)` | After retrieval, reranking, and deduplication | Annotating, reordering, or filtering results |
| `lifespan(client)` | Around the whole client lifetime | Owning connections, clients, background tasks |

Events are batch shaped. `IngestEvent` carries `documents` (a batch import arrives as one event with the whole batch) and `operation` (`"create"` or `"update"`). `DeleteEvent` carries the deleted `documents` in their last-known state, since the rows are already gone; a cascade arrives as one event. `SearchRequest` carries `query`, `filter`, `search_type`, and `limit`. `before_search` returns the request to search with, and may modify any of its fields. The query feeds both the vector and the full-text side. `after_search` returns the result list, and reads a request whose `search_type` is the one retrieval actually used: `hybrid` where a text search left it unset, `vector` for an image query. Hooks run in the order listed in config, each receiving the previous hook's output.

Hooks receive the `HaikuRAG` client, so they can search, read repositories, and store their own state.

## Registering a hook

```python
from haiku.rag.hooks import Hook

class AbbreviationHook(Hook):
    async def before_search(self, client, request):
        request.query = my_glossary.expand(request.query)
        return request

    async def after_search(self, client, request, results):
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

## Owning resources

Factories are called during client construction, before the database is open, so they must not acquire resources. Acquire them in `lifespan` instead, an async context manager around the client's lifetime:

```python
from contextlib import asynccontextmanager

import httpx

from haiku.rag.hooks import Hook

class GlossaryHook(Hook):
    @asynccontextmanager
    async def lifespan(self, client):
        async with httpx.AsyncClient() as http:
            self.http = http
            yield
```

Lifespans are entered in the order listed in config, once the store is open, and exited in reverse order while the store, embedder and reranker are all still usable.

Failing on entry fails `async with HaikuRAG(...)` and unwinds the lifespans already started: an activated hook that cannot start is a startup failure, not something to run degraded. Failing on exit is logged and swallowed, so one hook's teardown cannot strand another's. A hook is told which exception is being unwound, whether it came from the client's caller or from a later hook failing to start, but cannot suppress it.

## Background work

A hook that runs a background task owns stopping it. Cancel it on the way out, before anything awaits it:

```python
@asynccontextmanager
async def lifespan(self, client):
    async with asyncio.TaskGroup() as tg:
        task = tg.create_task(self.refresh_periodically(client))
        try:
            yield
        finally:
            task.cancel()
```

The `task.cancel()` is not optional. A `TaskGroup` that exits cleanly waits for its children instead of cancelling them, so an endless loop parked in one never returns and client shutdown hangs.

## Result annotations

`after_search` hooks can attach free-text notes on `SearchResult.annotations`. Annotations survive context expansion (merged results union the notes of their constituents, deduplicated) and render as `Note: text` lines in the agent-facing output used by the QA skills. MCP responses carry the field as part of the `SearchResult` model. This keeps the context cost proportional to what was retrieved instead of the size of your vocabulary.

## Semantics

- **Post-commit hooks are best-effort observers.** By the time `after_ingest` or `after_delete` runs, the operation has committed. A hook failure is logged, subsequent hooks still run, and the operation still returns success (the ingester proceeds through its normal success path). Correctness-critical derived state therefore needs its own retry or reconciliation, such as the backfill loop below. `before_search` and `after_search` failures propagate: nothing has committed and failing the search is visible to the caller.
- **Post-commit hooks are not a supported transformation point.** Mutating event models does not alter the committed record; explicit client writes are separate operations and are not atomic with the original write.
- **Update equals ingest.** `after_ingest` fires for both creation and content updates, with `event.operation` set to `"create"` or `"update"` (creation against an already-stored URI reports `"update"`). Treat both as "replace any state you derived from these documents". The operation is informational, for notification or sync hooks. Metadata-only and title-only updates do not fire.
- **Batch your writes.** A batch import delivers all its documents in one event. A hook keeping LanceDB state should write once per event, not once per document, to avoid creating a table version per document.
- **Hooks run after the write commits.** They execute outside the store's write lock, so a hook may itself write to the database, and a hook failure never rolls back the document write.
- **Rebuild does not fire hooks.** `rebuild` re-chunks and re-embeds but never changes document content, so content-derived state is unaffected.
- **Backfill is your loop.** A hook enabled on an existing database can backfill by iterating `client.list_documents()` and calling its own `after_ingest`.
- **State lives in the database.** Hooks may create their own LanceDB tables via `client.store`. Prefix table names with `hook_` so they never collide with core tables or future migrations. State then travels with the database and its backups.
