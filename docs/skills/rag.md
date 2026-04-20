# RAG Skill

The RAG skill is the primary way to use haiku.rag tools. It bundles search, document browsing, and citation management into a single skill with managed state.

## `create_skill(db_path?, config?)`

```python
from haiku.rag.skills.rag import create_skill

skill = create_skill(db_path=db_path, config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `None` | Path to LanceDB database. Falls back to `HAIKU_RAG_DB` env var, then config default. |
| `config` | `None` | `AppConfig` instance. If None, uses `get_config()`. |

## Tools

| Tool | Purpose |
|------|---------|
| `search(query, limit?)` | Hybrid search (vector + full-text) with context expansion |
| `list_documents()` | List all documents in the knowledge base |
| `get_document(query)` | Retrieve a document by ID, title, or URI |
| `cite(chunk_ids)` | Register chunk IDs as citations for the current answer |

## State

The skill manages a `RAGState` under the `"rag"` namespace:

```python
class RAGState(BaseModel):
    citation_index: dict[str, Citation] = {}
    citations: list[list[str]] = []
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = {}
```

- **citation_index** — All citations indexed by chunk ID (deduplicated across turns).
- **citations** — Per-turn lists of chunk IDs registered via the `cite` tool.
- **document_filter** — SQL WHERE clause applied to `search` and `list_documents` calls. Set this to scope queries to specific documents.
- **searches** — Search results keyed by query string.
