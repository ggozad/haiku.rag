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
# Serialize all namespaces
data = context.dump_namespaces()
# {"my_namespace": {"count": 0}}

# Restore a namespace from serialized data
context.load_namespace("my_namespace", MyState, data["my_namespace"])
```

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

**Tool: `ask(question, document_name?)`**

Runs the research graph in conversational mode and returns a `QAResult`. When a `ToolContext` is provided:

- Prior answers from `QASessionState.qa_history` are matched via embedding similarity
- The answer is appended to `qa_history`
- Background summarization is triggered
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

Toolsets are designed to be composed into custom pydantic-ai agents:

```python
from dataclasses import dataclass
from pydantic_ai import Agent
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.tools import (
    ToolContext,
    RAGDeps,
    create_search_toolset,
    create_qa_toolset,
    create_document_toolset,
)

# Toolsets are created once at configuration time
search = create_search_toolset(Config)
qa = create_qa_toolset(Config)
docs = create_document_toolset(Config)

@dataclass
class MyDeps:
    """Must satisfy the RAGDeps protocol (client + tool_context)."""
    client: HaikuRAG
    tool_context: ToolContext | None = None

agent = Agent(
    "openai:gpt-4o",
    deps_type=MyDeps,
    instructions="You are a helpful research assistant.",
    toolsets=[search, qa, docs],
)

async with HaikuRAG("path/to/db.lancedb") as client:
    context = ToolContext()
    deps = MyDeps(client=client, tool_context=context)

    result = await agent.run("What documents do we have about climate?", deps=deps)
    print(result.output)

    # Access accumulated state
    from haiku.rag.tools.search import SearchState, SEARCH_NAMESPACE
    search_state = context.get(SEARCH_NAMESPACE, SearchState)
    if search_state:
        print(f"Total search results: {len(search_state.results)}")
```

Tool functions access `client` and `tool_context` via pydantic-ai's `RunContext.deps`, so toolsets can be created once and reused across requests. Your deps type just needs to satisfy the `RAGDeps` protocol (have `client` and `tool_context` attributes).

All toolsets respect session-level document filters when a `SessionState` is registered in the context. This means setting `SessionState.document_filter` restricts all tools simultaneously.

## AG-UI State Management

When using the chat agent with [AG-UI](https://docs.ag-ui.com) streaming, `ChatDeps` implements the `StateHandler` protocol. State is emitted under a namespaced key via `state_key`:

```python
from haiku.rag.agents.chat import (
    AGUI_STATE_KEY, ChatDeps, create_chat_agent, prepare_chat_context,
)
from haiku.rag.tools import ToolContext, ToolContextCache

# Agent can be created once at startup
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
