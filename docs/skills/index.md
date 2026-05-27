# Skills

Skills put haiku.rag in front of a model. A skill bundles tools, an instruction prompt, and managed state into a unit that drops into any Pydantic AI agent via `SkillToolset`. haiku.rag ships two skills and supports custom skills.

Built on [haiku.skills](https://github.com/ggozad/haiku.skills).

## Available skills

| Skill | What it does | Reach for it when |
|-------|--------------|-------------------|
| [`rag`](rag.md) | Search, retrieve, and cite content from a knowledge base. | The model needs to find and quote evidence from documents. |
| [`rag-analysis`](analysis.md) | Same as `rag`, plus a sandboxed Python interpreter mounting every document as a virtual filesystem. | The question requires computation, aggregation, structural traversal, or section-scoped reading. |

To ship your own skill (bundled with its own database), see [Custom skills](custom.md).

## Your first agent

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

result = await agent.run("What does the knowledge base say about X?")
print(result.output)
```

The skill searches, cites, and answers. You supply the model and the question.

To run analysis against the same database, swap in the `rag-analysis` skill or attach both:

```python
from haiku.rag.skills.rag import create_skill as create_rag_skill
from haiku.rag.skills.analysis import create_skill as create_analysis_skill

rag = create_rag_skill(db_path="my.lancedb")
analysis = create_analysis_skill(db_path="my.lancedb")
toolset = SkillToolset(skills=[rag, analysis])
```

The agent reads each skill's description and routes questions itself. See the individual skill pages for the tool surface, state model, and worked examples.

## State

Each skill manages its own state under a dedicated namespace. State is synced via the AG-UI protocol when using `AGUIAdapter`.

```python
rag_state = toolset.get_namespace("rag")
analysis_state = toolset.get_namespace("analysis")
```

Both state models track citations, the current document filter, and per-turn searches. Analysis state also carries the sandbox execution log. See [RAG skill: state](rag.md#state) and [Analysis skill: state](analysis.md#state).

## Database path resolution

Both skills resolve the database path in the same order:

1. `db_path` argument passed to `create_skill()`
2. `HAIKU_RAG_DB` environment variable
3. Config default (`config.storage.data_dir / "haiku.rag.lancedb"`)

## AG-UI streaming for web apps

For browser apps, use pydantic-ai's `AGUIAdapter` to stream tool calls, text, and state deltas:

```python
from pydantic_ai.ui.ag_ui import AGUIAdapter

adapter = AGUIAdapter(agent=agent, run_input=run_input)
event_stream = adapter.run_stream()
sse_event_stream = adapter.encode_stream(event_stream)
```

See the [Web application](../apps.md) reference implementation.

## Exposing via MCP

To use a skill from Claude Desktop or another MCP-aware client, run the MCP server:

```bash
haiku-rag mcp --stdio
```

The server exposes the skill tools (search, ask, analyze) over MCP. See [MCP](../mcp.md).

## Discovery

Skills are registered as Python entry points under `haiku.skills`. They are discovered automatically:

```bash
haiku-skills list --use-entrypoints
# rag — Search, retrieve and analyze documents using RAG.
# rag-analysis — Analyze documents using code execution in a sandboxed interpreter.
```

This is what makes custom skills installable as plain pip packages. See [Custom skills](custom.md).
