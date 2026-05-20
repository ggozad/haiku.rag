# Analysis Skill

Plain RAG (search → cite → answer) works for questions whose answer sits in a chunk or two: "Who wrote this?", "What does X say about Y?". It struggles when the answer requires touching the whole corpus, reading a specific section in full, or doing arithmetic on the data.

The analysis skill (`rag-analysis`) gives the agent a second tool (`execute_code`) that runs Python in a sandboxed interpreter against a structured view of your documents. The agent can search, read, count, slice, and compare without leaving the tool call. Citations work the same way as the rag skill.

`client.analyze`, `haiku-rag analyze`, the MCP `analyze` tool, and the chat TUI (when `-s analysis` is enabled) all run through this skill.

## When to use it

Reach for the analysis skill when the question needs more than a search:

- **Aggregation across the corpus.** "How many documents mention security vulnerabilities?"
- **Section-scoped reading.** "Summarize Section 5 of paper Y."
- **Structural comparison.** "Do both papers have an Experimental Results section?"
- **Computation on retrieved data.** "What's the average revenue across these quarterly reports?"
- **Multi-step chains.** Search, filter the results in Python, search again, aggregate, all in one tool call.

For everyday Q&A, the [RAG skill](rag.md) is faster and cheaper. Attach both and the agent routes.

## How it works

Two things make the agent's programs short and the resulting analyses tractable:

1. **Search and document listing are awaitable inside the code.** `await search(query)` returns the same hits the rag skill sees: chunk IDs, text, source metadata, picture refs. The agent can immediately filter, sort, count, or follow up with another search without exiting the tool call.

2. **Every document is mounted as a virtual filesystem at `/documents/{id}/`.** The agent reads four files per document: identifiers and metadata, full text, a list of structured items (paragraphs, tables, figures, headings), and a section tree built from the document's headings. The structure exposes what search alone hides. The agent can navigate from a search hit to the section it lives in, slice a single section instead of pulling the whole document, or scan a document's text directly when keyword precision matters.

A search hit is always a starting point. The agent reads structure around it, drills into the right section, and cites the chunks it actually used. Chunk IDs from search results and chunk IDs surfaced through the VFS are both accepted by `cite`.

### Sandbox guarantees

The interpreter is [pydantic-monty](https://github.com/pydantic/monty), isolated from the host:

- **Virtual filesystem only.** `/documents/` is the entire FS.
- **No network.** HTTP, sockets, and the `requests` family are unavailable.
- **Limited imports.** Only `json`, `re`, `math`, `pathlib`.
- **Execution timeout** (default 60s, configurable via `analysis.code_timeout`).
- **Output truncation** (default 50000 chars, configurable via `analysis.max_output_chars`).

Variables persist between `execute_code` calls within one invocation, so the agent can build state step by step. A fresh sandbox is built per `client.analyze` call.

## Tools

| Tool | Purpose |
|------|---------|
| `search(query, limit?)` | Hybrid search with context expansion. Same as the RAG skill's `search`. |
| `execute_code(code)` | Run Python in a sandboxed interpreter with VFS access. |
| `cite(chunk_ids)` | Register chunk IDs as citations. Call before producing the final answer. |

`list_documents` isn't exposed as a top-level tool but is available inside `execute_code` as `await list_documents()`.

## State

`AnalysisState` lives under the `"analysis"` namespace:

```python
class AnalysisState(BaseModel):
    document_filter: str | None = None
    executions: list[CodeExecutionEntry] = []
    citation_index: dict[str, Citation] = {}
    citations: list[str] = []
    searches: dict[str, list[SearchResult]] = {}
```

- **document_filter** — SQL WHERE clause applied to `search` and the VFS. The LLM can't bypass it: both views are scoped.
- **executions** — Each `execute_code` call appends an entry with code, stdout, stderr, success. Cleared at the start of each invocation.
- **citation_index** — All citations indexed by chunk ID. Accumulates across invocations.
- **citations** — Chunk IDs registered via `cite` during the current invocation. Deduplicated, cleared per-invocation.
- **searches** — Search results from both the `search` tool and sandbox-internal searches. Cleared per-invocation.

## `create_skill(db_path?, config?)`

```python
from haiku.rag.skills.analysis import create_skill

skill = create_skill(db_path="my.lancedb")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `None` | Path to LanceDB database. Falls back to `HAIKU_RAG_DB` env var, then config default. |
| `config` | `None` | `AppConfig` instance. Falls back to `get_config()`. |

## Use it

### From `client.analyze`

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG("my.lancedb") as client:
    result = await client.analyze("How many documents mention 'security'?")
    print(result.answer)
    for citation in result.citations:
        print(citation.uri, citation.title)
```

`client.analyze` runs the skill end-to-end and returns an `AnalysisResult` with `answer` and `citations`. The executed Python programs live on `AnalysisState.executions` during the run, not on the returned result.

### Combine with the RAG skill

```python
from haiku.rag.skills.rag import create_skill as create_rag_skill
from haiku.rag.skills.analysis import create_skill as create_analysis_skill
from haiku.skills.agent import SkillToolset

rag = create_rag_skill(db_path="my.lancedb")
analysis = create_analysis_skill(db_path="my.lancedb")
toolset = SkillToolset(skills=[rag, analysis])
```

The agent routes Q&A to the rag skill and computational questions to rag-analysis.

## What the agent actually writes

You don't write these programs yourself. The agent does, inside `execute_code`. Seeing the shape helps when you tune prompts, debug a run via `AnalysisState.executions`, or design a custom skill.

**Aggregate across the corpus.** *"How many documents mention security vulnerabilities?"*

```python
hits = await search("security vulnerability", limit=50)

doc_ids = {h['document_id'] for h in hits}
print(f"{len(doc_ids)} documents mention security vulnerabilities")

# Cite the top hit per document
seen = set()
for hit in hits:
    if hit['document_id'] not in seen:
        seen.add(hit['document_id'])
        await cite(hit['chunk_id'])
```

**Read one section in depth.** *"Summarize Section 5."*

```python
from pathlib import Path
import json

doc_id = "..."  # from a prior search or list_documents
toc = json.loads(Path(f'/documents/{doc_id}/toc.json').read_text())

section = next(n for n in toc['tree'] if n['title'].startswith('5'))
start, end = section['item_range']
lines = Path(f'/documents/{doc_id}/items.jsonl').read_text().splitlines()[start:end]

for line in lines:
    print(json.loads(line)['text'])

await cite(section['chunk_ids'])
```

The section node already aggregates the chunks underneath it, so the agent cites the whole section without a separate search.

**Compare structure across documents.** *"Do both papers have an Experimental Results section?"*

```python
from pathlib import Path
import json

for doc_id in ["doc-a-id", "doc-b-id"]:
    toc = json.loads(Path(f'/documents/{doc_id}/toc.json').read_text())
    print(f"\n=== {toc['title']} ===")
    for node in toc['tree']:
        if 'experiment' in node['title'].lower():
            print(f"  {node['title']} (pages {node['page_numbers']})")
            await cite(node['chunk_ids'])
```

## Context filter

The `filter` parameter is enforced at the deps layer. The LLM can't bypass it: both the VFS and search results are scoped to the filter.

```python
result = await client.analyze(
    "Summarize all findings",
    filter="uri LIKE '%confidential%'"
)
```

Useful for scoping to a corpus subset, enforcing access control, or restricting context.

## Configuration

```yaml
analysis:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
  code_timeout: 60.0      # Max seconds per code execution
  max_output_chars: 50000 # Truncate output after this many chars
```

When `analysis.model` is unset, the skill falls back to `qa.model`.

See [Search and question answering](../configuration/qa.md#analysis-configuration) for the full set.
