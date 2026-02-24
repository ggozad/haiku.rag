# Skills

haiku.rag exposes its RAG capabilities as [haiku.skills](https://github.com/ggozad/haiku.skills) skills. Skills are self-contained units that bundle tools, instructions, and state — they can be composed into any pydantic-ai agent via `SkillToolset`.

## Available Skills

| Skill | Description |
|-------|-------------|
| [`rag`](rag.md) | Search, retrieve, and answer questions from the knowledge base |
| [`rag-rlm`](rlm.md) | Computational analysis via code execution |

## Discovery

Skills are registered as Python entrypoints under `haiku.skills`. They are discovered automatically by `haiku.skills`:

```bash
haiku-skills list --use-entrypoints
# rag — Search, retrieve and analyze documents using RAG.
# rag-rlm — Analyze documents using code execution in a sandboxed interpreter.
```

## Usage

```python
from haiku.rag.skills.rag import create_skill
from haiku.skills.agent import SkillToolset
from pydantic_ai import Agent

skill = create_skill(db_path=db_path, config=config)
toolset = SkillToolset(skills=[skill])

agent = Agent(
    "openai:gpt-4o",
    instructions=toolset.system_prompt,
    toolsets=[toolset],
)

result = await agent.run("What documents do we have?")
```

## Database Path Resolution

Both skills resolve the database path in the same order:

1. `db_path` argument passed to `create_skill()`
2. `HAIKU_RAG_DB` environment variable
3. Config default (`config.storage.data_dir / "haiku.rag.lancedb"`)

## State Management

Each skill manages its own state under a dedicated namespace. State is automatically synced via the AG-UI protocol when using `AGUIAdapter`.

```python
rag_state = toolset.get_namespace("rag")
rlm_state = toolset.get_namespace("rlm")
```

See the individual skill pages for state model details.

## AG-UI Streaming

For web applications, use pydantic-ai's `AGUIAdapter` to stream tool calls, text, and state deltas:

```python
from pydantic_ai.ag_ui import AGUIAdapter

adapter = AGUIAdapter(agent=agent, run_input=run_input)
event_stream = adapter.run_stream()
sse_event_stream = adapter.encode_stream(event_stream)
```

See the [Web Application](../apps.md#web-application) for a complete implementation.
