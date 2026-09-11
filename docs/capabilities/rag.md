# RAG Capability

`RAGCapability` adds grounded document search, sandboxed Python over the documents, and citations to a Pydantic AI agent. It is deferred by default, so its instructions and tools do not consume model context until loaded.

## Tools

| Tool | Purpose |
|---|---|
| `search(query, limit?)` | Hybrid vector and full-text search with context expansion. |
| `execute_code(code)` | Run Python against the virtual document filesystem. |
| `cite(chunk_ids)` | Register retrieved or filesystem-derived chunk IDs as answer citations. |

The sandbox exposes documents under `/documents/{document_id}/` with `metadata.json`, `content.txt`, `items.jsonl`, `chunks.jsonl` (chunk ids with their metadata) and `toc.json`. In code, `await search()` results carry `chunk_meta` and `await list_documents()` rows carry `metadata`. The interpreter's limits and the per-call budgets are listed under [MCP, Code](../mcp.md#code). The sandbox opens only when the model first executes code, so a question answered from search alone never pays for it.

## Create and compose

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

`create_capability` accepts `db_path`, `config`, `defer_loading`, `request_limit`, `sources`, and `vision`. `sources` names the configured databases the capability covers, the sandbox filesystem included, all of them when omitted (see [Database selection](index.md#database-selection)). Set `defer_loading=False` for a dedicated agent where routing is unnecessary. The default request limit is 30 model requests per question; set `request_limit=None` to disable it. `vision` controls whether picture results are attached to search returns as images and should reflect the model the hosting agent runs; it defaults to the configured QA model's `vision` flag.

When the limit is reached, `search` and `execute_code` are removed while `cite` remains for two further requests that call one of the capability's tools, so the model can register citations before answering from evidence already gathered. Requests spent on other tools do not count against that window. Unrelated agent and capability tools remain available. A new agent run starts a fresh limit, so multi-turn chat does not consume one shared budget.

When `qa.max_searches` or `qa.max_executions` runs out, the exhausted tool keeps failing rather than disappearing, and the instructions name it on every following request. Searching from inside `execute_code` does not count against `qa.max_searches`.

For the high-level convenience API:

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("my.lancedb") as client:
    answer, citations = await client.ask("Which quarter had the highest revenue?")
    print(answer)
```

## State

When agent dependencies expose a `state` dictionary, the capability maintains a `RAGState` under `"rag"`:

```python
class RAGState(BaseModel):
    citation_index: dict[str, Citation]
    citations: list[str]
    document_filter: str | None
    sources: list[str] | None
    evidence: CapabilityEvidenceRecord
    searches: dict[str, list[SearchResult]]
    executions: list[CodeExecutionEntry]
```

`document_filter`, `sources`, `citation_index` and `evidence` persist across runs. Citations, searches and executions are cleared when a new question starts; a run that resumes a question keeps the evidence it is still answering from.

`evidence` records which chunks this capability retrieved and cited, and in which question. `haiku.rag.capabilities.ledger.citation_status(records, question=...)` derives `missing`, `grounded` or `ungrounded` from it.

State is ordinary application state; the capability does not depend on AG-UI. An AG-UI application can expose it using Pydantic AI's standard adapter.

## Context management

This capability does not alter the message history. To stop long conversations resending old retrieved content, register the [compaction capability](compaction.md) alongside it.

## Domain context and vision

`prompts.domain_preamble` is prepended to the packaged capability instructions. When the capability's `vision` gate is on (by default, when the configured QA model has `vision: true`), picture results are attached to search returns as `BinaryContent`.

See [Search and question answering](../configuration/qa.md) and [picture processing](../configuration/processing.md#picture-handling).
