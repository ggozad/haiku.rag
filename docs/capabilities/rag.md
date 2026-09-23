# RAG capability

`RAGCapability` adds grounded document search, sandboxed Python over the documents, and citations to a Pydantic AI agent.

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[rag(db_path="my.lancedb")],
)

result = await agent.run("What safety equipment does the manual require?")
print(result.output)
```

`HaikuRAG.ask` runs the same capability behind a single call. To add evidence compaction and the citation policy, see [Compose an agent](index.md#compose-an-agent).

## Tools

| Tool | Purpose |
|---|---|
| `search(query, limit?)` | Hybrid vector and full-text search with context expansion. |
| `execute_code(code)` | Run Python against the virtual document filesystem. |
| `cite(chunk_ids)` | Register retrieved or filesystem-derived chunk IDs as answer citations. |

## Parameters

`create_capability` takes:

| Parameter | Default | Meaning |
|---|---|---|
| `db_path` | `None` | A database path, where the configuration places none. See [Database selection](index.md#database-selection) |
| `config` | the loaded configuration | An `AppConfig` |
| `defer_loading` | `False` | Keep the instructions and tools out of the model context until the model loads the capability |
| `rag` | `None` | An open `HaikuRAG` client to use. The capability never closes it |
| `request_limit` | `30` | Model requests per agent run. `None` disables it |
| `sources` | all configured | The databases the capability covers, the sandbox filesystem included |
| `vision` | `qa.model.vision` | Whether picture results are attached to search returns as images. Set it for the model the agent runs |

## Budgets

When the request limit is reached, `search` and `execute_code` are removed and `cite` remains for two further requests that call one of the capability's tools, so the model can register citations before answering. Requests spent on other tools do not count against that window, and other tools stay available. Each agent run starts a fresh limit.

When `qa.max_searches` or `qa.max_executions` runs out, the exhausted tool keeps failing rather than disappearing, and the instructions name it on every following request. Searching from inside `execute_code` does not count against `qa.max_searches`. The budgets are described under [Search and question answering](../configuration/qa.md#question-answering-configuration).

## Sandbox

`execute_code` runs Python in a [Monty](https://github.com/pydantic/monty) interpreter over a virtual filesystem. Each document the question covers is mounted at `/documents/{document_id}/`:

| File | Contents |
|---|---|
| `metadata.json` | Document id, title, URI, creation time and metadata |
| `content.txt` | The full text |
| `items.jsonl` | Document items in reading order |
| `chunks.jsonl` | Chunk ids with their metadata |
| `toc.json` | The section tree |

Code can also `await search(query)`, whose rows carry `chunk_meta` and `picture_refs`, and `await list_documents()`, whose rows carry `metadata`.

Monty is a Python subset. Useful modules include `json`, `re`, `math`, `pathlib`, `datetime`, `collections`, `itertools`, `functools` and `dataclasses`. `decimal` and `statistics` are absent. Class inheritance, generators, `match` statements and iterating a file object are not supported. Files are read-only, and there is no network and no filesystem beyond `/documents`.

One sandbox serves every `execute_code` call of a run, so variables persist between calls. It opens when the model first executes code. `sandbox.code_timeout` and `sandbox.max_output_chars` bound each call, see [Sandbox configuration](../configuration/qa.md#sandbox-configuration). The MCP server's `execute_code` tool runs the same sandbox, one program per call.

## State

When agent dependencies expose a `state` dictionary, the capability keeps a `RAGState` under `"rag"`:

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

`document_filter`, `sources`, `citation_index` and `evidence` persist across runs. Citations, searches and executions are cleared when a new question starts. A run that resumes a question keeps the evidence it is still answering from.

`evidence` records which chunks this capability cited, in which question, and whether the citing question also retrieved them. `haiku.rag.capabilities.ledger.citation_status(records, question=...)` derives `missing`, `grounded` or `ungrounded` from it.

The capability does not alter the message history. To stop long conversations resending old retrieved content, register the [compaction capability](compaction.md).

## Domain context and vision

`prompts.domain_preamble` is prepended to the packaged capability instructions, see [Prompts](../configuration/prompts.md). With `vision` on, picture results are attached to search returns as `BinaryContent`. See [picture handling](../configuration/processing.md#picture-handling).
