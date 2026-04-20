# Analysis Skill

The analysis skill provides computational analysis via code execution. It writes and runs Python code in a sandboxed interpreter to answer questions that require computation, aggregation, or data traversal.

## `create_skill(db_path?, config?)`

```python
from haiku.rag.skills.analysis import create_skill

skill = create_skill(db_path=db_path, config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `None` | Path to LanceDB database. Falls back to `HAIKU_RAG_DB` env var, then config default. |
| `config` | `None` | `AppConfig` instance. If None, uses `get_config()`. |

## Tools

| Tool | Purpose |
|------|---------|
| `search(query, limit?)` | Hybrid search (vector + full-text) with context expansion |
| `list_documents()` | List all documents in the knowledge base |
| `execute_code(code)` | Execute Python code in a sandboxed interpreter with VFS access |
| `cite(chunk_ids)` | Register chunk IDs as citations for the current answer |

## State

The skill manages an `AnalysisState` under the `"analysis"` namespace:

```python
class AnalysisState(BaseModel):
    document_filter: str | None = None
    executions: list[CodeExecutionEntry] = []
    citation_index: dict[str, Citation] = {}
    citations: list[list[str]] = []
    searches: dict[str, list[SearchResult]] = {}
```

- **document_filter** — SQL WHERE clause applied to `search` and `list_documents` calls.
- **executions** — Each `execute_code` call appends a `CodeExecutionEntry` with code, stdout, stderr, and success status.
- **citation_index** / **citations** — Same per-turn citation tracking as the RAG skill.
- **searches** — Search results from both the `search` tool and sandbox-internal searches.

## Usage with RAG Skill

Combine both skills to give the agent full RAG + analysis capabilities:

```python
from haiku.rag.skills.rag import create_skill as create_rag_skill
from haiku.rag.skills.analysis import create_skill as create_analysis_skill
from haiku.skills.agent import SkillToolset
from haiku.skills.prompts import build_system_prompt
from pydantic_ai import Agent

rag = create_rag_skill(db_path=db_path)
analysis = create_analysis_skill(db_path=db_path)
toolset = SkillToolset(skills=[rag, analysis])

agent = Agent(
    "openai:gpt-4o",
    instructions=build_system_prompt(toolset.skill_catalog),
    toolsets=[toolset],
)
```

See the [Analysis Agent](../agents/analysis.md) documentation for details on how the underlying sandbox works.
