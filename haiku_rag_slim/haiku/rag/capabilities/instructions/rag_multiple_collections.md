
## Multiple collections

The corpus spans multiple collections, and each interface names them differently:

- Search results carry a `Collection:` line naming the collection each came from.
- In code, `await search(...)` and `await list_documents()` return `source`, the
  collection an item came from.
- The mounted files do not. `/documents/{id}/metadata.json` has no `source`, so
  map ids to collections with `await list_documents()` before reading the
  filesystem per collection.

### Choosing the collections to search

`search` takes `sources`, a list of collection names. It narrows that one search;
the collections this run covers stay the same.

- When the user names collections, or results you already have show which
  collection holds the answer, pass those names.
- When you do not know which collection holds the answer, omit `sources` and
  search them all.
- A collection the user names can be the wrong one. If a narrowed search does
  not answer the question, your next search omits `sources`. Do not retry other
  wordings in the same collections first.
- Never report that the knowledge base lacks the information until a search
  without `sources` has failed to find it.
- A name outside this run's collections fails the call and lists the names
  available. Use the names exactly as listed.

Group, count and compare by `source` when the question is about collections
rather than about documents.
