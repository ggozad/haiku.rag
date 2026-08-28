
## Multiple collections

The corpus spans multiple collections, and each interface names them differently:

- Results from searches spanning multiple collections carry a `Collection:` line.
- In code, `await search(...)` and `await list_documents()` return `source`, the
  collection an item came from.
- The mounted files do not. `/documents/{id}/metadata.json` has no `source`, so
  map ids to collections with `await list_documents()` before reading the
  filesystem per collection.

Group, count and compare by `source` when the question is about collections
rather than about documents.
