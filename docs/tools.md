# Tools & Skills

haiku.rag exposes its RAG capabilities through a [haiku.skills](https://github.com/ggozad/haiku.skills) skill. The skill provides tools for search, Q&A, analysis, and research that can be composed into any pydantic-ai agent via `SkillToolset`.

For lower-level access, `haiku.rag.tools` provides individual `FunctionToolset` factories used internally by agents.

## RAG Skill

The RAG skill is the primary way to use haiku.rag tools. It bundles all capabilities into a single skill with managed state.

```python
from haiku.rag.skills.rag import create_skill
from haiku.skills.agent import SkillToolset
from pydantic_ai import Agent

skill = create_skill(db_path=db_path, config=config)
toolset = SkillToolset(skills=[skill])

agent = Agent(
    "openai:gpt-4o",
    instructions=toolset.system_prompt,
    toolsets=[toolset],
)

result = await agent.run("What documents do we have?")
```

### `create_skill(db_path?, config?)`

Creates a RAG skill instance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `None` | Path to LanceDB database. Falls back to `HAIKU_RAG_DB` env var, then config default. |
| `config` | `None` | `AppConfig` instance. If None, uses `get_config()`. |

### Tools

| Tool | Purpose |
|------|---------|
| `search(query, limit?)` | Hybrid search (vector + full-text) with context expansion |
| `list_documents(limit?, offset?, filter?)` | Paginated document listing |
| `get_document(query)` | Retrieve a document by ID, title, or URI |
| `ask(question)` | Q&A with citations via the QA agent |
| `analyze(question, document?, filter?)` | Computational analysis via code execution (requires Docker) |
| `research(question)` | Deep multi-agent research producing comprehensive reports |
| `get_session_context(query)` | Retrieve relevant prior Q&A from the session |

### State

The skill manages a `RAGState` under the `"rag"` namespace:

```python
class RAGState(BaseModel):
    citations: list[Any] = []
    qa_history: list[QAHistoryEntry] = []
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = {}
    documents: list[DocumentInfo] = []
    reports: list[ResearchEntry] = []
```

State is automatically synced via the AG-UI protocol when using `AGUIAdapter`. Access it programmatically:

```python
rag_state = toolset.get_namespace("rag")
if rag_state:
    print(f"Citations: {len(rag_state.citations)}")
    print(f"Q&A history: {len(rag_state.qa_history)}")
```

### AG-UI Streaming

For web applications, use pydantic-ai's `AGUIAdapter` to stream tool calls, text, and state deltas:

```python
from pydantic_ai.ag_ui import AGUIAdapter

adapter = AGUIAdapter(agent=agent, run_input=run_input)
event_stream = adapter.run_stream()
sse_event_stream = adapter.encode_stream(event_stream)
```

See the [Web Application](apps.md#web-application) for a complete implementation.

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
