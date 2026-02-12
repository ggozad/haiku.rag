# Toolsets

haiku.rag provides composable `FunctionToolset` factories in `haiku.rag.tools`. Each factory creates a pydantic-ai `FunctionToolset` that can be mixed into any agent. A shared `ToolContext` lets toolsets accumulate state (search results, citations, QA history) across invocations.

## ToolContext

`ToolContext` is a namespace-based state container. Toolsets register Pydantic models under string namespaces, and any toolset sharing the same context can read or write the same state.

```python
from haiku.rag.tools import ToolContext

context = ToolContext()
```

### Registering and accessing state

```python
from pydantic import BaseModel

class MyState(BaseModel):
    count: int = 0

context.register("my_namespace", MyState())

# Get state (returns None if not registered)
state = context.get("my_namespace")

# Get with type checking (returns None if wrong type)
state = context.get("my_namespace", MyState)

# Get or create (creates default if not registered)
state = context.get_or_create("my_namespace", MyState)
```

### Serialization

The entire context can be serialized and restored:

```python
# Serialize all namespaces (keyed by namespace)
data = context.dump_namespaces()
# {"my_namespace": {"count": 0}}

# Restore a namespace from serialized data
context.load_namespace("my_namespace", MyState, data["my_namespace"])
```

For AG-UI state management, use flat snapshots:

```python
# Flat snapshot of all namespaces (for AG-UI state)
snapshot = context.build_state_snapshot()
# {"document_filter": [], "citations": [], "citation_registry": {}, "qa_history": []}

# Restore from flat snapshot (updates registered namespaces in place)
context.restore_state_snapshot(snapshot)
```

### Preparing context for toolsets

`prepare_context()` registers the required namespaces for a given set of features:

```python
from haiku.rag.tools import ToolContext, prepare_context

context = ToolContext()
prepare_context(context, features=["search", "qa"], state_key="my_app")
```

This is idempotent and registers `SessionState` (for search, QA, and analysis features) and `QASessionState` (for QA). The chat agent's `prepare_chat_context()` is a thin wrapper that defaults to chat features and sets the AG-UI state key.

## Search Toolset

`create_search_toolset()` provides hybrid search (vector + full-text) with context expansion and citation tracking.

```python
from haiku.rag.tools import create_search_toolset

search = create_search_toolset(config)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | AppConfig |
| `expand_context` | `True` | Expand results with surrounding chunks |
| `base_filter` | `None` | SQL WHERE clause applied to all searches |
| `tool_name` | `"search"` | Name of the tool exposed to the agent |

**Tool: `search(query, limit?, filter?)`**

Searches the knowledge base and returns formatted results. When a `ToolContext` with `SessionState` is registered, citations get stable indices via `citation_registry`.

**State:** Search results accumulate in `SearchState.results` under the `haiku.rag.search` namespace.

## Document Toolset

`create_document_toolset()` provides document browsing, retrieval, and summarization.

```python
from haiku.rag.tools import create_document_toolset

docs = create_document_toolset(config)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | AppConfig (used for summarization LLM) |
| `base_filter` | `None` | SQL WHERE clause for list operations |

**Tools:**

- `list_documents(page?)` — Paginated document listing (50 per page). Returns `DocumentListResponse` with document titles, URIs, and pagination info.
- `get_document(query)` — Retrieve a document by title or URI. Uses `find_document()` which tries exact URI match, then partial URI match, then partial title match.
- `summarize_document(query)` — Generate an LLM summary of a document's content.

## QA Toolset

`create_qa_toolset()` provides question answering via the research graph, with prior answer recall and background summarization.

```python
from haiku.rag.tools import create_qa_toolset

qa = create_qa_toolset(config)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | AppConfig |
| `base_filter` | `None` | SQL WHERE clause applied to searches |
| `tool_name` | `"ask"` | Name of the tool exposed to the agent |
| `on_ask_complete` | `None` | Callback `(QASessionState, AppConfig) -> None` invoked after each QA cycle |

**Tool: `ask(question, document_name?)`**

Runs the research graph in conversational mode and returns a `QAResult`. When a `ToolContext` is provided:

- Prior answers from `QASessionState.qa_history` are matched via embedding similarity
- The answer is appended to `qa_history`
- `on_ask_complete` callback is invoked (if provided)
- Citations get stable indices via `SessionState.citation_registry`

**State:** QA history accumulates in `QASessionState` under the `haiku.rag.qa_session` namespace.

### Using `run_qa_core()` directly

For programmatic use without an agent, `run_qa_core()` provides the same QA flow:

```python
from haiku.rag.tools.qa import run_qa_core

result = await run_qa_core(
    client=client,
    config=config,
    question="What are the main features?",
    document_name="User Guide",       # optional document filter
    context=context,                   # optional ToolContext
    session_context="User is building a web app",  # optional
    on_qa_complete=my_callback,        # optional post-QA callback
)

print(result.answer)
print(result.confidence)
for citation in result.citations:
    print(f"  [{citation.index}] {citation.document_title}")
```

## Analysis Toolset

`create_analysis_toolset()` provides computational analysis via the RLM agent, which writes and executes Python code in a Docker sandbox.

```python
from haiku.rag.tools import create_analysis_toolset

analysis = create_analysis_toolset(config)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | AppConfig |
| `base_filter` | `None` | SQL WHERE clause applied to searches |
| `tool_name` | `"analyze"` | Name of the tool exposed to the agent |

**Tool: `analyze(task, document_name?)`**

Executes a computational task via code execution and returns an `AnalysisResult`. Requires Docker — see [RLM Agent](agents/rlm.md) for setup.

## Composing Custom Agents

Toolsets are designed to be composed into custom pydantic-ai agents. Use `AgentDeps` and `prepare_context` for minimal boilerplate:

```python
from pydantic_ai import Agent
from haiku.rag.client import HaikuRAG
from haiku.rag.tools import (
    AgentDeps,
    ToolContext,
    prepare_context,
    create_search_toolset,
    create_qa_toolset,
    create_document_toolset,
)

# Toolsets are created once at configuration time
search = create_search_toolset(config)
qa = create_qa_toolset(config)
docs = create_document_toolset(config)

agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDeps,
    instructions="You are a helpful research assistant.",
    toolsets=[search, qa, docs],
)

async with HaikuRAG("path/to/db.lancedb") as client:
    context = ToolContext()
    prepare_context(context, features=["search", "documents", "qa"])
    deps = AgentDeps(client=client, tool_context=context)

    result = await agent.run("What documents do we have about climate?", deps=deps)
    print(result.output)

    # Access accumulated state
    from haiku.rag.tools.search import SearchState, SEARCH_NAMESPACE
    search_state = context.get(SEARCH_NAMESPACE, SearchState)
    if search_state:
        print(f"Total search results: {len(search_state.results)}")
```

`AgentDeps` satisfies the `RAGDeps` protocol and implements the AG-UI state protocol (`state` getter/setter). For AG-UI streaming, pass a `state_key`:

```python
deps = AgentDeps(client=client, tool_context=context, state_key="my_app")
```

Tool functions access `client` and `tool_context` via pydantic-ai's `RunContext.deps`, so toolsets can be created once and reused across requests.

All toolsets respect session-level document filters when a `SessionState` is registered in the context. This means setting `SessionState.document_filter` restricts all tools simultaneously.

## AG-UI State Management

Both `AgentDeps` and `ChatDeps` implement the AG-UI `StateHandler` protocol. State is emitted under a namespaced key via `state_key`.

**Custom agents** use `AgentDeps` + `prepare_context`:

```python
from haiku.rag.tools import AgentDeps, ToolContext, ToolContextCache, prepare_context

context = ToolContext()
prepare_context(context, features=["search", "qa"], state_key="my_app")
deps = AgentDeps(client=client, tool_context=context, state_key="my_app")
```

**Chat agent** uses `ChatDeps` + `prepare_chat_context` (adds chat-specific overrides like background summarization and initial context handling):

```python
from haiku.rag.agents.chat import (
    AGUI_STATE_KEY, ChatDeps, create_chat_agent, prepare_chat_context,
)
from haiku.rag.tools import ToolContext, ToolContextCache

agent = create_chat_agent(config)

# For multi-session apps, cache ToolContext per thread
cache = ToolContextCache()
context, _is_new = cache.get_or_create(thread_id)
prepare_chat_context(context)  # idempotent namespace registration

deps = ChatDeps(
    config=config,
    client=client,
    tool_context=context,
    state_key=AGUI_STATE_KEY,  # "haiku.rag.chat"
)
```

The emitted state structure:

```json
{
  "haiku.rag.chat": {
    "citations": [],
    "qa_history": [],
    "session_context": null,
    "document_filter": [],
    "citation_registry": {}
  }
}
```

State flows bidirectionally — the frontend sends its current state on each request, and the agent emits deltas (JSON Patch) reflecting server-side updates (new citations, QA history entries, session context). The server always prefers its own `session_context` over the client's value, since background summarization may have updated it between requests. See the [Web Application](apps.md#web-application) for a complete implementation.

## Filter Helpers

`haiku.rag.tools.filters` provides utilities for building SQL filters:

**`build_document_filter(document_name)`** — Builds a LIKE filter matching against both `uri` and `title`, case-insensitive. Also matches without spaces (e.g., "TB MED 593" matches "tbmed593").

**`build_multi_document_filter(document_names)`** — Combines multiple document name filters with OR logic.

**`combine_filters(filter1, filter2)`** — Combines two filters with AND logic. Returns `None` if both are `None`.

**`get_session_filter(context, base_filter?)`** — Extracts `document_filter` from `SessionState` in the `ToolContext`, builds a SQL filter from it, and combines with an optional `base_filter`.
