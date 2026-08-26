
## Multiple databases

The corpus spans several databases, and each interface names them differently:

- `analysis_search` results carry a `Database:` line.
- In code, `await search(...)` and `await list_documents()` return `source`, the
  configured database an item came from.
- The mounted files do not. `/documents/{id}/metadata.json` has no `source`, so
  map ids to databases with `await list_documents()` before reading the
  filesystem per database.

Group, count and compare by `source` when the question is about databases rather
than about documents.
