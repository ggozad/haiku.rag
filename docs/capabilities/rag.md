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

`create_capability` accepts `db_path`, `config`, `defer_loading`, `request_limit`, and `vision`. Set `defer_loading=False` for a dedicated RAG agent where routing is unnecessary. The default request limit is 20 model requests per question; set `request_limit=None` to disable it. `vision` controls whether picture results are attached to search returns as images and should reflect the model the hosting agent runs; it defaults to the configured QA model's `vision` flag.

When the limit is reached, `rag_search` is removed while `rag_cite` remains for two further requests, so the model can register citations before answering from evidence already gathered. Unrelated agent and capability tools remain available. A new agent run starts a fresh limit, so multi-turn chat does not consume one shared budget.

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

`prompts.domain_preamble` is prepended to the packaged capability instructions. When the capability's `vision` gate is on (by default, when the configured QA model has `vision: true`), picture results are attached to search returns as `BinaryContent`.

See [Search and question answering](../configuration/qa.md) and [picture processing](../configuration/processing.md#picture-handling).
