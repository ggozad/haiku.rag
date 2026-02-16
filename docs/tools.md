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

## Tool Prompts

`build_tools_prompt()` generates system prompt guidance for your toolsets — when to use each tool, the `document_name` parameter pattern, and usage examples. It's designed to be spliced into any agent's instructions alongside your own domain-specific guidance.

```python
from haiku.rag.tools import build_tools_prompt

# Generate guidance for the toolsets you're using
tools_prompt = build_tools_prompt(["search", "qa", "documents"])
```

Combine it with your own instructions:

```python
from pydantic_ai import Agent
from haiku.rag.tools import AgentDeps, build_tools_prompt

tools_prompt = build_tools_prompt(["search", "qa"])

agent = Agent(
    "anthropic:claude-sonnet-4-5-20250929",
    deps_type=AgentDeps,
    instructions=f"""You are a medical research assistant.
{tools_prompt}

You also have access to:
- "check_interactions" - Use when the user asks about drug interactions.""",
    toolsets=[search_toolset, qa_toolset, my_custom_toolset],
)
```

Available features: `"search"`, `"qa"`, `"documents"`, `"analysis"`.

## Composing Custom Agents

### Using `build_toolkit` (recommended)

`build_toolkit()` bundles toolsets, prompt, and context creation for a given feature set:

```python
from pydantic_ai import Agent
from haiku.rag.client import HaikuRAG
from haiku.rag.tools import AgentDeps, build_toolkit

toolkit = build_toolkit(config, features=["search", "documents", "qa"])

agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDeps,
    instructions=f"You are a helpful research assistant.\n{toolkit.prompt}",
    toolsets=toolkit.toolsets,
)

async with HaikuRAG("path/to/db.lancedb") as client:
    context = toolkit.create_context()
    deps = AgentDeps(client=client, tool_context=context)

    result = await agent.run("What documents do we have about climate?", deps=deps)
    print(result.output)

    # Access accumulated state
    from haiku.rag.tools.search import SearchState, SEARCH_NAMESPACE
    search_state = context.get(SEARCH_NAMESPACE, SearchState)
    if search_state:
        print(f"Total search results: {len(search_state.results)}")
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | AppConfig |
| `features` | `["search", "documents"]` | Features to enable |
| `base_filter` | `None` | SQL WHERE clause applied to all toolsets |
| `expand_context` | `True` | Expand search results with surrounding chunks |
| `on_qa_complete` | `None` | Callback invoked after each QA cycle |

**`Toolkit` properties:**

- `toolsets` — list of `FunctionToolset` instances to pass to the Agent
- `prompt` — tool guidance text for the system prompt
- `features` — the feature list this toolkit was built from
- `create_context(state_key=None)` — create a prepared `ToolContext` matching these features
- `prepare(context, state_key=None)` — register namespaces on an existing `ToolContext`

### Using individual factories

For full control, create toolsets individually with `create_*_toolset()`, `build_tools_prompt()`, and `prepare_context()`:

```python
from pydantic_ai import Agent
from haiku.rag.client import HaikuRAG
from haiku.rag.tools import (
    AgentDeps,
    ToolContext,
    build_tools_prompt,
    prepare_context,
    create_search_toolset,
    create_qa_toolset,
    create_document_toolset,
)

search = create_search_toolset(config)
qa = create_qa_toolset(config)
docs = create_document_toolset(config)

features = ["search", "documents", "qa"]
tools_prompt = build_tools_prompt(features)

agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDeps,
    instructions=f"You are a helpful research assistant.\n{tools_prompt}",
    toolsets=[search, qa, docs],
)

async with HaikuRAG("path/to/db.lancedb") as client:
    context = ToolContext()
    prepare_context(context, features=features)
    deps = AgentDeps(client=client, tool_context=context)

    result = await agent.run("What documents do we have about climate?", deps=deps)
    print(result.output)
```

`AgentDeps` satisfies the `RAGDeps` protocol and implements the AG-UI state protocol (`state` getter/setter). For AG-UI streaming, set `state_key` on the `ToolContext` (via `prepare_context` or `toolkit.create_context`):

```python
context = toolkit.create_context(state_key="my_app")
deps = AgentDeps(client=client, tool_context=context)
```

Tool functions access `client` and `tool_context` via pydantic-ai's `RunContext.deps`, so toolsets can be created once and reused across requests.

For complete runnable examples, see [`examples/custom_agent.py`](https://github.com/ggozad/haiku.rag/tree/main/examples/custom_agent.py) (standalone) and [`examples/custom_agent_agui.py`](https://github.com/ggozad/haiku.rag/tree/main/examples/custom_agent_agui.py) (AG-UI streaming server).

All toolsets respect session-level document filters when a `SessionState` is registered in the context. This means setting `SessionState.document_filter` restricts all tools simultaneously.

## AG-UI State Management

Both `AgentDeps` and `ChatDeps` implement the AG-UI `StateHandler` protocol. `ChatDeps` extends `AgentDeps` with chat-specific config and state handling. State is emitted under a namespaced key via `state_key` on the `ToolContext` — set it once via `prepare_context()`.

**Custom agents** use `AgentDeps` + `prepare_context`:

```python
from haiku.rag.tools import AgentDeps, ToolContext, ToolContextCache, prepare_context

context = ToolContext()
prepare_context(context, features=["search", "qa"], state_key="my_app")
deps = AgentDeps(client=client, tool_context=context)
```

**Chat agent** uses `ChatDeps` + `build_chat_toolkit` (adds chat-specific defaults like background summarization):

```python
from haiku.rag.agents.chat import (
    AGUI_STATE_KEY, ChatDeps, build_chat_toolkit, create_chat_agent,
)
from haiku.rag.tools import ToolContextCache

chat_toolkit = build_chat_toolkit(config)
agent = create_chat_agent(config, toolkit=chat_toolkit)

# For multi-session apps, cache ToolContext per thread
cache = ToolContextCache()
context, is_new = cache.get_or_create(thread_id)
if is_new:
    chat_toolkit.prepare(context, state_key=AGUI_STATE_KEY)

deps = ChatDeps(
    config=config,
    client=client,
    tool_context=context,
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
