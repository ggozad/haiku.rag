# Capabilities

haiku.rag provides two native [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/):

| Capability | Use it for |
|---|---|
| [`RAGCapability`](rag.md) | Grounded document search and citations. |
| [`AnalysisCapability`](analysis.md) | Corpus computation and structural analysis with sandboxed Python. |

Both capabilities are deferred by default. An agent initially sees only their descriptions and the standard `load_capability` tool. Instructions and tools enter the model context only when the model loads a capability.

## Compose an agent

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.rag import create_capability

rag = create_capability(db_path="my.lancedb")
agent = Agent("openai:gpt-5", capabilities=[rag])

result = await agent.run("What does the knowledge base say about X?")
print(result.output)
```

Attach both capabilities when an agent should choose between retrieval and computation:

```python
from haiku.rag.capabilities.analysis import create_capability as analysis
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[rag(db_path="my.lancedb"), analysis(db_path="my.lancedb")],
)
```

## State

Capabilities use a plain `state: dict[str, Any]` attribute on agent dependencies when one is available. RAG state lives under `"rag"`; analysis state lives under `"analysis"`. This keeps state independent of any transport or UI protocol.

Applications serving AG-UI should adapt the agent with Pydantic AI's `AGUIAdapter`. Native model and tool events require no haiku.rag-specific bridge.

## Database path

Both factories resolve their database in this order:

1. The `db_path` argument.
2. `HAIKU_RAG_DB`.
3. `config.storage.data_dir / "haiku.rag.lancedb"`.
