# RLM Skill

The RLM (Recursive Language Model) skill provides computational analysis via code execution. It writes and runs Python code in a sandboxed interpreter to answer questions that require computation, aggregation, or data traversal.

## `create_skill(db_path?, config?)`

```python
from haiku.rag.skills.rlm import create_skill

skill = create_skill(db_path=db_path, config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `None` | Path to LanceDB database. Falls back to `HAIKU_RAG_DB` env var, then config default. |
| `config` | `None` | `AppConfig` instance. If None, uses `get_config()`. |

## Tools

| Tool | Purpose |
|------|---------|
| `analyze(question, document?, filter?)` | Answer analytical questions using code execution |

**Parameters:**

- `question` — The analytical question to answer.
- `document` — Optional document ID or title to pre-load for analysis.
- `filter` — Optional SQL WHERE clause to filter documents.

## State

The skill manages an `RLMState` under the `"rlm"` namespace:

```python
class RLMState(BaseModel):
    analyses: list[AnalysisEntry] = []

class AnalysisEntry(BaseModel):
    question: str
    answer: str
    program: str | None = None
```

Each `analyze` call appends an `AnalysisEntry` with the question, answer, and executed program.

## Usage with RAG Skill

Combine both skills to give the agent full RAG + analysis capabilities:

```python
from haiku.rag.skills.rag import create_skill as create_rag_skill
from haiku.rag.skills.rlm import create_skill as create_rlm_skill
from haiku.skills.agent import SkillToolset
from pydantic_ai import Agent

rag = create_rag_skill(db_path=db_path)
rlm = create_rlm_skill(db_path=db_path)
toolset = SkillToolset(skills=[rag, rlm])

agent = Agent(
    "openai:gpt-4o",
    instructions=toolset.system_prompt,
    toolsets=[toolset],
)
```

See the [RLM Agent](../agents/rlm.md) documentation for details on how the underlying agent works.
