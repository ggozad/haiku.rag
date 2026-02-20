# RAG Skill

The RAG skill is the primary way to use haiku.rag tools. It bundles search, Q&A, document browsing, and research into a single skill with managed state.

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
| `list_documents(limit?, offset?, filter?)` | Paginated document listing |
| `get_document(query)` | Retrieve a document by ID, title, or URI |
| `ask(question)` | Q&A with citations via the QA agent |
| `research(question)` | Deep multi-agent research producing comprehensive reports |

## State

The skill manages a `RAGState` under the `"rag"` namespace:

```python
class RAGState(BaseModel):
    citations: list[Citation] = []
    qa_history: list[QAHistoryEntry] = []
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = {}
    documents: list[DocumentInfo] = []
    reports: list[ResearchEntry] = []
```

- **citations** — Accumulated citations from `ask` calls, with sequential indexing across calls.
- **qa_history** — Questions and answers from `ask` calls. Prior Q&A is used as context for follow-up questions when embeddings are similar.
- **document_filter** — SQL WHERE clause applied to `search`, `ask`, and `research` calls. Set this to scope queries to specific documents.
- **searches** — Search results keyed by query string.
- **documents** — Documents seen via `list_documents` or `get_document` (deduplicated by ID).
- **reports** — Research reports from `research` calls.
