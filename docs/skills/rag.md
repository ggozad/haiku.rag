# RAG Skill

The `rag` skill answers questions over a knowledge base with hybrid search, structure-aware context expansion, and explicit citations. `client.ask`, `haiku-rag ask`, the MCP `ask_question` tool, and the chat TUI all run through this skill.

## When to use it

- The model needs to find and quote evidence from a document corpus.
- You want citations under every answer.
- You're building a Q&A agent, a documentation chatbot, or any RAG-style integration.

If the question requires *computation* over the corpus (counts, aggregates, comparisons, section-scoped reading), reach for the [Analysis skill](analysis.md) instead, or attach both.

## Tools

| Tool | Purpose |
|------|---------|
| `search(query, limit?)` | Hybrid search (vector + full-text) with section-aware context expansion. Returns `chunk_id`, content, `doc_item_refs`, `picture_refs`, `picture_captions`, source metadata. |
| `list_documents()` | List all documents in the knowledge base. |
| `get_document(query)` | Fetch a document by ID, title, or URI. Partial matches work. |
| `cite(chunk_ids)` | Register chunk IDs as citations for the current answer. The agent calls this before writing the final response. |

## State

The skill manages a `RAGState` under the `"rag"` namespace:

```python
class RAGState(BaseModel):
    citation_index: dict[str, Citation] = {}
    citations: list[str] = []
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = {}
```

- **citation_index** — All citations indexed by chunk ID. Accumulates across invocations so historical chunk IDs stay resolvable in UI scrollback.
- **citations** — Chunk IDs registered via `cite` during the current invocation. Deduplicated, cleared at the start of each invocation.
- **document_filter** — SQL WHERE clause applied to `search` and `list_documents`. Persists across invocations.
- **searches** — Search results keyed by query string. Cleared at the start of each invocation.

## `create_skill(db_path?, config?)`

```python
from haiku.rag.skills.rag import create_skill

skill = create_skill(db_path="my.lancedb")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `None` | Path to LanceDB database. Falls back to `HAIKU_RAG_DB` env var, then config default. |
| `config` | `None` | `AppConfig` instance. Falls back to `get_config()`. |

## Examples

### Minimal agent

```python
from haiku.rag.skills.rag import create_skill
from haiku.skills.agent import SkillToolset
from haiku.skills.prompts import build_system_prompt
from pydantic_ai import Agent

rag = create_skill(db_path="my.lancedb")
toolset = SkillToolset(skills=[rag])

agent = Agent(
    "openai-chat:gpt-4o",
    instructions=build_system_prompt(toolset.skill_catalog),
    toolsets=[toolset],
)

result = await agent.run("What does the manual say about safety procedures?")
print(result.output)

# Inspect what the model cited
state = toolset.get_namespace("rag")
for chunk_id in state.citations:
    citation = state.citation_index[chunk_id]
    print(f"- {citation.document_title}: {citation.content[:100]}…")
```

### Domain customization

Set a domain preamble in `haiku.rag.yaml` and the skill picks it up:

```yaml
prompts:
  domain_preamble: |
    The knowledge base contains the operations manual for the Helios solar array.
    "The array" or unqualified specs refer to Helios. Terminology like "string"
    refers to a series-connected panel chain, not text.
```

To scope a session to a subset of documents, set the filter on the namespace state:

```python
state = toolset.get_namespace("rag")
state.document_filter = "uri LIKE '%helios/v4/%'"

result = await agent.run("What's the maintenance interval for the inverters?")
```

The filter applies to every `search` and `list_documents` call for the rest of the session, including the model can't bypass it from inside.

### Combining with the analysis skill

Attach both skills and the agent routes between them:

```python
from haiku.rag.skills.rag import create_skill as create_rag_skill
from haiku.rag.skills.analysis import create_skill as create_analysis_skill

rag = create_rag_skill(db_path="my.lancedb")
analysis = create_analysis_skill(db_path="my.lancedb")
toolset = SkillToolset(skills=[rag, analysis])

agent = Agent(
    "openai-chat:gpt-4o",
    instructions=build_system_prompt(toolset.skill_catalog),
    toolsets=[toolset],
)

# Q&A → uses rag
await agent.run("What safety equipment is required on-site?")

# Computational question → uses rag-analysis
await agent.run("How many checklists mention torque specifications?")
```

### Streaming to a web frontend

Wrap the agent with `AGUIAdapter` to stream tool calls, text deltas, and state changes to a CopilotKit-style frontend:

```python
from pydantic_ai.ui.ag_ui import AGUIAdapter

adapter = AGUIAdapter(agent=agent, run_input=run_input)
sse_stream = adapter.encode_stream(adapter.run_stream())
```

See the [Web application](../apps.md) reference implementation for the full Starlette + Next.js setup.

### Exposing via MCP

To call the skill from Claude Desktop (or any MCP client), run the MCP server:

```bash
haiku-rag serve --mcp --stdio
```

The exposed `ask_question` tool runs this skill. See [MCP](../mcp.md) for the configuration block.

## Configuration

The skill picks up its model and search behavior from the standard config sections:

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: true
    temperature: 0.3
    vision: false           # set true for vision-capable QA models
  max_searches: 3

search:
  limit: 5
  max_context_chars: 10000
```

See [Search and question answering](../configuration/qa.md) for every knob.

## Vision support

When `qa.model.vision: true` is set, the skill's `search` tool attaches picture bytes to its tool returns as `BinaryContent`. The model can then read figures, diagrams, and screenshots directly alongside the surrounding text. Requires `processing.pictures != none` so the bytes exist on disk. See the [pictures × embedder × QA model matrix](../configuration/processing.md#picture-handling) for the combinations that make sense.

## Customizing the skill prompt

The skill's instruction prompt lives in `SKILL.md` inside the package. For behavior changes (different phrasing, refusal style, additional rules), the supported path is to fork the skill with `haiku-rag create-skill` and edit the generated `SKILL.md`. The `domain_preamble` field above is for *what the corpus is about*, not for *how the agent should behave*. See [Custom skills](custom.md).
