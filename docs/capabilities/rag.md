# RAG Capability

`RAGCapability` adds grounded document search and citations to a Pydantic AI agent. It is deferred by default, so its instructions and tools do not consume model context until loaded.

## Tools

| Tool | Purpose |
|---|---|
| `rag_search(query, limit?)` | Hybrid vector and full-text search with context expansion. |
| `rag_cite(chunk_ids)` | Register exact result chunk IDs as answer citations. |

The distinct `rag_` prefix lets this capability coexist with analysis and other search providers.

## Create and compose

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.rag import create_capability

rag = create_capability(db_path="my.lancedb")
agent = Agent("openai:gpt-5", capabilities=[rag])

result = await agent.run("What safety equipment does the manual require?")
print(result.output)
```

`create_capability` accepts `db_path`, `config`, and `defer_loading`. Set `defer_loading=False` for a dedicated RAG agent where routing is unnecessary.

## State

When agent dependencies expose a `state` dictionary, the capability maintains a `RAGState` under `"rag"`:

```python
class RAGState(BaseModel):
    citation_index: dict[str, Citation]
    citations: list[str]
    document_filter: str | None
    searches: dict[str, list[SearchResult]]
```

`document_filter` persists between runs. Current citations and searches reset for each run, while the citation index remains available to the host application.

State is ordinary application state; the capability does not depend on AG-UI. An AG-UI application can expose it using Pydantic AI's standard adapter.

## Context management

Large RAG tool results from earlier user turns are replaced with a short marker before model requests. Tool-call pairing and current-turn evidence are retained. This prevents long conversations from repeatedly sending old retrieved content.

## Domain context and vision

`prompts.domain_preamble` is prepended to the packaged capability instructions. When the selected QA model has `vision: true`, picture results are attached to search returns as `BinaryContent`.

See [Search and question answering](../configuration/qa.md) and [picture processing](../configuration/processing.md#picture-handling).
