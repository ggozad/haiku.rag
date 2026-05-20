# Custom Skills

The two skills haiku.rag ships work against any LanceDB database. When you want a *domain-specific* skill that bundles its own data, prompt, and tool surface (for example, a "recipes" skill that knows about cooking and ships with a recipes database), generate one with `haiku-rag create-skill`.

The generated package is a regular pip-installable Python package that registers as a `haiku.skills` entry point. Any haiku.skills-aware host (haiku.skills CLI, your own agent, the AG-UI adapter) discovers it automatically.

## When to use a custom skill

- The model should consult a specific knowledge base for a specific kind of question, alongside other skills.
- You want a different instruction prompt than the generic `rag` skill (different tone, refusal style, domain rules).
- You want to ship a knowledge base plus its prompt as one distributable unit.
- You're running multiple skills against different databases in the same agent.

If you just want to point a haiku.rag database at your own model and prompt, configure `haiku.rag.yaml` and use the built-in `rag` skill. No custom package needed.

## Generate

```bash
haiku-rag create-skill \
  --name recipes \
  --db /path/to/recipes.lancedb \
  --tools search,cite \
  --description "Recipe and cooking knowledge base" \
  --preamble "You are a culinary expert helping with recipes and cooking techniques."
```

Then install and use:

```bash
uv pip install -e ./recipes-skill

haiku-skills list --use-entrypoints
# recipes — Recipe and cooking knowledge base

haiku-skills chat --use-entrypoints --skill recipes
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Skill name (lowercase alphanumeric and hyphens). Required. | — |
| `--db` | Path to the LanceDB database to embed. Required. | — |
| `--description` | One-line skill description. The agent reads this to decide when to invoke. | Standard RAG description |
| `--tools` | Comma-separated tool subset, or `all`. | `all` |
| `--preamble` | Custom preamble for the skill's instructions. | Standard RAG preamble |
| `--config-file` | Path to a `haiku.rag.yaml` to embed alongside the database. | None |
| `--output` / `-o` | Output directory. | Current directory |

### Available tools

`cite`, `execute_code`, `get_document`, `list_documents`, `search`.

Drop `execute_code` from `--tools` if the skill shouldn't run sandboxed Python. That gives you a search-and-cite-only skill with no analysis capabilities.

## Anatomy of a generated skill

```
{name}-skill/
├── pyproject.toml
└── {name}_skill/
    ├── __init__.py    # create_skill() entry point
    ├── SKILL.md       # Skill metadata and instructions
    └── assets/
        ├── {name}.lancedb/   # The embedded database
        └── haiku.rag.yaml    # Optional config (only if --config-file passed)
```

- **`SKILL.md`** carries the instruction prompt the agent will follow. The frontmatter includes the skill name and description. Everything below is the prompt body. Edit this to change behavior.
- **`__init__.py`** exposes `create_skill()` (the entry point) and `visualize_chunk()` for rendering visual grounding.
- **`assets/{name}.lancedb/`** is the database, shipped inside the package.
- **`assets/haiku.rag.yaml`** (optional) pins provider settings the skill needs.

The package can be installed locally with `uv pip install -e .` or published to PyPI.

## Generating visual grounding from a custom skill

Each generated skill exposes a `visualize_chunk()` function that returns the chunk's bounding boxes rendered onto its source page:

```python
from recipes_skill import visualize_chunk

images = await visualize_chunk(chunk_id)
# images is a list of PIL.Image objects, one per page the chunk covers
images[0].save("citation.png")
```

Pass chunk IDs from skill citations or search results. Same prerequisites as elsewhere in haiku.rag: documents need stored page images, and the chunk must come from a PDF or other docling-converted source.

## Multi-skill agents

Each generated skill is self-contained with its own database and instructions. Compose multiple skills in one agent and the model routes between them via their descriptions:

```python
from recipes_skill import create_skill as create_recipes_skill
from medic_skill import create_skill as create_medic_skill
from haiku.skills.agent import SkillToolset
from haiku.skills.prompts import build_system_prompt
from pydantic_ai import Agent

recipes = create_recipes_skill()
medic = create_medic_skill()
toolset = SkillToolset(skills=[recipes, medic])

agent = Agent(
    "openai-chat:gpt-4o",
    instructions=build_system_prompt(toolset.skill_catalog),
    toolsets=[toolset],
)

await agent.run("What's the optimal temperature for braising short ribs?")
# Routes to recipes

await agent.run("What's the field treatment for tension pneumothorax?")
# Routes to medic
```

Each skill maintains state under its own namespace (`recipes`, `medic`, …), so citations and searches don't collide.

## Writing a skill from scratch

`create-skill` is the convenience path. If you need full control over the tools, state model, or instruction loading, write the skill against [haiku.skills](https://github.com/ggozad/haiku.skills) directly. The generated package in `{name}_skill/__init__.py` is a good reference. It composes haiku.rag's `_tools` factory with a `haiku.skills.Skill` and registers under the `haiku.skills` entry point group in `pyproject.toml`.

See the haiku.skills repository for the full Skill contract.
