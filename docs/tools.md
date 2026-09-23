# Toolsets

For agent integrations, use the native Pydantic AI [capabilities](capabilities/index.md). `haiku.rag.tools` provides lower-level `FunctionToolset` factories for building custom agents.

## Low-Level Toolsets

### RAGDeps Protocol

All toolsets read their client from the agent dependencies through the `RAGDeps` protocol, which requires a `client: HaikuRAG` attribute:

```python
from dataclasses import dataclass

from haiku.rag.client import HaikuRAG


@dataclass
class MyDeps:
    client: HaikuRAG
```

### Search Toolset

`create_search_toolset()` provides hybrid search with context expansion.

```python
from haiku.rag.tools import create_search_toolset

search = create_search_toolset(config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | `AppConfig` |
| `expand_context` | `True` | Expand results with surrounding chunks |
| `base_filter` | `None` | SQL WHERE clause applied to all searches |
| `tool_name` | `"search"` | Name of the tool exposed to the agent |
| `on_results` | `None` | Callback `(list[SearchResult]) -> None` invoked with results |
| `max_searches` | `None` | Searches allowed per run; past it the tool fails and tells the agent to answer |

Picture results are attached as images when `config.qa.model.vision` is set.

### Document Toolset

`create_document_toolset()` provides document browsing and retrieval.

```python
from haiku.rag.tools import create_document_toolset

docs = create_document_toolset(config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | `AppConfig` |
| `base_filter` | `None` | SQL WHERE clause for list operations |

**Tools:**

- `list_documents(page?)` — Paginated document listing (50 per page).
- `get_document(query)` — Retrieve a document by title or URI.
- `summarize_document(query)` — Generate an LLM summary of a document's content.

## Filter Helpers

`haiku.rag.tools.filters` provides utilities for building SQL filters:

- **`build_multi_document_filter(document_names)`** — Combines multiple document name filters with OR logic. Matches against both `uri` and `title`, case-insensitive, with and without spaces. Returns `None` for an empty list.
- **`build_document_id_filter(document_ids)`** — Matches exactly the given document ids. Returns `None` for an empty list.
