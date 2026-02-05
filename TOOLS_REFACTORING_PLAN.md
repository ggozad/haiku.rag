# Tools Extraction Refactoring Plan

## Goal

Extract tools from haiku.rag agents into a reusable `tools/` module, enabling users to create pydantic-ai agents outside haiku.rag and compose toolsets as needed.

## Target API

```python
from pydantic_ai import Agent
from haiku.rag import HaikuRAG
from haiku.rag.tools import ToolContext, create_search_toolset, create_document_toolset

async with HaikuRAG(db_path) as client:
    context = ToolContext()
    search_tools = create_search_toolset(client, config, context)
    doc_tools = create_document_toolset(client, config, context)

    agent = Agent(
        'anthropic:claude-sonnet',
        toolsets=[search_tools, doc_tools]
    )
    result = await agent.run("Find documents about X")

    # Access accumulated state after run
    search_state = context.get("haiku.rag.search")
    for result in search_state.results:
        print(f"{result.document_title}")
```

## Design Principles

1. **ToolContext is a pure generic container** - No special-cased fields. Toolsets register their own Pydantic model state under namespaces.

2. **Shared state via same namespace** - Multiple toolsets can share state (e.g., citations, filters) by registering under the same namespace.

3. **App manages identity** - ToolContext has no session/user identity. The app layer manages `session_id -> ToolContext` mapping.

4. **Toolsets are stateless factories** - `create_*_toolset()` returns a `FunctionToolset`. State lives in the context they're given.

## ToolContext Design

```python
class ToolContext(BaseModel):
    """Generic state container for toolsets.

    Toolsets register Pydantic model state under namespaces.
    Multiple toolsets can share state via the same namespace.
    """
    _namespaces: dict[str, BaseModel] = PrivateAttr(default_factory=dict)

    def register(self, namespace: str, state: BaseModel) -> None: ...
    def get(self, namespace: str) -> BaseModel | None: ...
    def get_or_create(self, namespace: str, factory: Callable[[], T]) -> T: ...
    def clear_namespace(self, namespace: str) -> None: ...
    def clear_all(self) -> None: ...
    def dump_namespaces(self) -> dict[str, dict[str, Any]]: ...
    def load_namespace(self, namespace: str, state_type: type[T], data: dict) -> T: ...
```

## Toolset State Examples

Each toolset defines its own state model:

```python
# Search toolset state
class SearchState(BaseModel):
    results: list[SearchResult] = []
    filter: str | None = None

SEARCH_NAMESPACE = "haiku.rag.search"

# QA toolset state
class QAState(BaseModel):
    history: list[QAResult] = []

QA_NAMESPACE = "haiku.rag.qa"

# Shared citation state (used by multiple toolsets)
class CitationState(BaseModel):
    registry: dict[str, int] = {}

    def get_or_assign_index(self, chunk_id: str) -> int:
        if chunk_id in self.registry:
            return self.registry[chunk_id]
        new_index = len(self.registry) + 1
        self.registry[chunk_id] = new_index
        return new_index

CITATION_NAMESPACE = "haiku.rag.citations"
```

## Multi-User/Session Management

App layer manages context routing:

```python
# App maintains context per session
contexts: dict[str, ToolContext] = {}

def get_context(session_id: str) -> ToolContext:
    if session_id not in contexts:
        contexts[session_id] = ToolContext()
    return contexts[session_id]

# When running agent
context = get_context(user_session_id)
toolsets = [create_search_toolset(client, config, context)]
await agent.run(prompt, toolsets=toolsets)
```

## New Module Structure

```
haiku_rag_slim/haiku/rag/
├── tools/                      # NEW
│   ├── __init__.py             # Public exports
│   ├── context.py              # ToolContext (generic state container)
│   ├── models.py               # QAResult, AnalysisResult
│   ├── filters.py              # build_document_filter, combine_filters
│   ├── search.py               # create_search_toolset()
│   ├── document.py             # create_document_toolset()
│   ├── qa.py                   # create_qa_toolset()
│   └── analysis.py             # create_analysis_toolset()
├── agents/                     # REFACTORED to use tools/
```

## Implementation Chunks

### Chunk 1: Create tools module foundation ✅ DONE
- Created `tools/__init__.py`, `tools/context.py`, `tools/models.py`, `tools/filters.py`
- Created `ToolContext` as generic namespace-based Pydantic model
- Moved filter utilities from `agents/chat/state.py` to `tools/filters.py`
- Created result models (`QAResult`, `AnalysisResult`)
- Added tests for ToolContext and filters

### Chunk 2: Create SearchToolset ✅ DONE
- Created `tools/search.py` with `create_search_toolset()`
- Defined `SearchState` model for accumulating search results
- Core search logic: `client.search()` → `client.expand_context()` → `format_for_agent()`
- Results accumulated in `SearchState` under `SEARCH_NAMESPACE`
- Added 13 tests for SearchToolset

### Chunk 3: Refactor QA Agent to use SearchToolset ✅ DONE
- Updated `agents/qa/agent.py` to use `create_search_toolset()`
- Added `base_filter` and `tool_name` parameters to `create_search_toolset()`
- QA agent now uses ToolContext + SearchState for result accumulation
- Public interface (`answer(question, filter)`) unchanged
- All 5 QA tests pass

### Chunk 4: Create DocumentToolset ✅ DONE
- Created `tools/document.py` with `create_document_toolset()`
- Defined `DocumentState`, `DocumentInfo`, `DocumentListResponse` models
- Extracted `list_documents`, `get_document`, `summarize_document` tools
- Moved `find_document` helper (now public)
- Added 13 tests

### Chunk 5: Create QAToolset ✅ DONE
- Created `tools/qa.py` with `create_qa_toolset()`
- Defined `QAState` model (tracks QA history)
- Runs research graph, returns structured `QAResult`
- Supports `base_filter`, `tool_name`, `session_context`, `prior_answers` params
- Added 7 tests

### Chunk 6: Create AnalysisToolset ✅ DONE
- Created `tools/analysis.py` with `create_analysis_toolset()`
- Defined `AnalysisState` model (tracks CodeExecution history)
- Extracted `analyze` tool (RLM delegation with filter support)
- Fixed circular import by using direct submodule imports
- Added 6 tests

### Chunk 7: Refactor Chat Agent ✅ DONE
- Removed `analyze` tool from chat agent (kept hardcoded, not composing toolsets)
- Reverted system prompt to pre-analyze version
- Removed `test_analyze_tool` test and cassette file
- All 47 chat agent tests pass

### Chunk 8: Refactor Research Graph
- Update `_search_one_step_logic` to use search toolset
- Verify research tests pass

### Chunk 9: Public API and Documentation
- Export from `haiku.rag.tools` and `haiku.rag`
- Update CLAUDE.md
- Add usage examples

## Verification

- Run `pytest` after each chunk
- Run `ty check` and `ruff check`
- Test with existing agents (QA, Chat, Research)
- Test with external agent using new toolsets

## Critical Files

- `haiku_rag_slim/haiku/rag/agents/chat/agent.py` - largest tool collection
- `haiku_rag_slim/haiku/rag/agents/qa/agent.py` - simplest, good starting point
- `haiku_rag_slim/haiku/rag/agents/chat/state.py` - filter utilities (now moved)
- `haiku_rag_slim/haiku/rag/agents/research/graph.py` - search tool inside step
- `haiku_rag_slim/haiku/rag/store/models/chunk.py` - SearchResult.format_for_agent()
