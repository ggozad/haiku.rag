# Analysis Capability

`AnalysisCapability` adds search, citations, and sandboxed Python computation over the document corpus. Use it for counts, aggregation, comparison, structural traversal, and section-scoped reading.

It is deferred by default, keeping its substantial instructions and tool schemas out of context until the model chooses to load it.

## Tools

| Tool | Purpose |
|---|---|
| `analysis_search(query, limit?)` | Search the corpus for evidence. |
| `analysis_execute_code(code)` | Run Python against the virtual document filesystem. |
| `analysis_cite(chunk_ids)` | Register retrieved or filesystem-derived chunk IDs. |

The sandbox exposes documents under `/documents/{document_id}/` with `metadata.json`, `content.txt`, `items.jsonl`, and `toc.json`.

## Compose with RAG

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.analysis import create_capability as analysis
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[
        rag(db_path="my.lancedb"),
        analysis(db_path="my.lancedb"),
    ],
)
```

For the high-level convenience API:

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("my.lancedb") as client:
    result = await client.analyze("Which quarter had the highest revenue?")
    print(result.answer)
```

## State

When dependencies expose a state dictionary, `AnalysisState` is stored under `"analysis"`. It contains the document filter, code execution log, searches, and citations. Per-run searches and executions reset automatically; the filter and citation index persist.

The capability lazily opens both LanceDB and the sandbox only after it is loaded and a tool requires them. Resources close at the end of the agent run.
