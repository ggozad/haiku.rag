# RAG Capability

`RAGCapability` adds grounded document search and citations to a Pydantic AI agent. It is deferred by default, so its instructions and tools do not consume model context until loaded.

## Tools

| Tool | Purpose |
|---|---|
| `rag_search(query, limit?)` | Hybrid vector and full-text search with context expansion. |
| `rag_cite(chunk_ids)` | Register exact result chunk IDs as answer citations. |

The distinct `rag_` prefix lets this capability coexist with analysis and other search providers.

## Create and compose

Register it on its own rather than alongside `AnalysisCapability`, which searches and
cites as well. See [Capabilities](index.md#compose-an-agent).

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[
        rag(db_path="my.lancedb"),
        compaction(),
        citation_policy(),
    ],
)

result = await agent.run("What safety equipment does the manual require?")
print(result.output)
```

`create_capability` accepts `db_path`, `config`, `defer_loading`, `request_limit`, and `vision`. Set `defer_loading=False` for a dedicated RAG agent where routing is unnecessary. The default request limit is 20 model requests per question; set `request_limit=None` to disable it. `vision` controls whether picture results are attached to search returns as images and should reflect the model the hosting agent runs; it defaults to the configured QA model's `vision` flag.

When the limit is reached, `rag_search` is removed while `rag_cite` remains for two further requests that call a RAG tool, so the model can register citations before answering from evidence already gathered. Requests spent on other capabilities do not count against that window. Unrelated agent and capability tools remain available. A new agent run starts a fresh limit, so multi-turn chat does not consume one shared budget.

## State

When agent dependencies expose a `state` dictionary, the capability maintains a `RAGState` under `"rag"`:

```python
class RAGState(BaseModel):
    citation_index: dict[str, Citation]
    citations: list[str]
    document_filter: str | None
    evidence: CapabilityEvidenceRecord
    searches: dict[str, list[SearchResult]]
```

`document_filter`, `citation_index` and `evidence` persist across runs. Citations and searches are cleared when a new question starts; a run that resumes a question keeps the evidence it is still answering from.

`evidence` records which chunks this capability retrieved and cited, and in which question. `haiku.rag.capabilities.ledger.citation_status(records, question=...)` derives `missing`, `grounded` or `ungrounded` from it, across capabilities.

State is ordinary application state; the capability does not depend on AG-UI. An AG-UI application can expose it using Pydantic AI's standard adapter.

## Context management

This capability does not alter the message history. To stop long conversations resending old retrieved content, register the [compaction capability](compaction.md) alongside it.

## Domain context and vision

`prompts.domain_preamble` is prepended to the packaged capability instructions. When the capability's `vision` gate is on (by default, when the configured QA model has `vision: true`), picture results are attached to search returns as `BinaryContent`.

See [Search and question answering](../configuration/qa.md) and [picture processing](../configuration/processing.md#picture-handling).
