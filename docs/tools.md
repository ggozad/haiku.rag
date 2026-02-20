# Toolsets

haiku.rag exposes its RAG capabilities as [haiku.skills](https://github.com/ggozad/haiku.skills) skills. See the [Skills](skills/index.md) section for the primary way to use haiku.rag tools.

For lower-level access, `haiku.rag.tools` provides individual `FunctionToolset` factories used internally by agents.

## Low-Level Toolsets

For advanced use cases, individual toolset factories are available in `haiku.rag.tools`. These are used internally by the QA agent and can be composed into custom agents.

### RAGDeps Protocol

All toolsets use the `RAGDeps` protocol for dependency injection:

```python
from haiku.rag.tools import RAGDeps

class MyDeps:
    def __init__(self, client: HaikuRAG):
        self.client = client
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

### Analysis Toolset

`create_analysis_toolset()` provides computational analysis via the RLM agent (Docker sandbox).

```python
from haiku.rag.tools import create_analysis_toolset

analysis = create_analysis_toolset(config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | `AppConfig` |
| `base_filter` | `None` | SQL WHERE clause applied to searches |
| `tool_name` | `"analyze"` | Name of the tool exposed to the agent |

## Filter Helpers

`haiku.rag.tools.filters` provides utilities for building SQL filters:

- **`build_document_filter(document_name)`** — Builds a LIKE filter matching against both `uri` and `title`, case-insensitive. Also matches without spaces (e.g., "TB MED 593" matches "tbmed593").
- **`build_multi_document_filter(document_names)`** — Combines multiple document name filters with OR logic.
- **`combine_filters(filter1, filter2)`** — Combines two filters with AND logic. Returns `None` if both are `None`.
