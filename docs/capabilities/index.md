# Capabilities

haiku.rag provides native [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/):

| Capability | Use it for |
|---|---|
| [`RAGCapability`](rag.md) | Grounded document search and citations. |
| [`AnalysisCapability`](analysis.md) | Corpus computation and structural analysis with sandboxed Python. |
| `EvidenceCompactionCapability` | Optional. Shrinking a conversation's history to the evidence that was cited. |

The two evidence capabilities are deferred by default. An agent initially sees only their descriptions and the standard `load_capability` tool. Instructions and tools enter the model context only when the model loads a capability.

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

## Multi-turn conversations

Every question adds its search results to the history, so requests grow turn after
turn, and can degrade answers or exceed a provider's limits as they do. Register the
compaction capability to replace earlier questions' evidence with the evidence that
was actually cited:

```python
from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[rag(db_path="my.lancedb"), compaction()],
)
```

Cited text and cited page images are kept in full, grouped by the question that
cited them, and stay citable by the same chunk ids. Everything else earlier becomes a
short receipt. Registering the capability is the only switch: leave it out and the
transcript reaches the model untouched. There is nothing to configure.

Compaction rewrites the request, never the stored history, so `all_messages()` still
holds everything the run gathered. Retained evidence still grows with the
conversation — this reduces what a request carries, it does not bound it. A host that
needs more aggressive pruning can compact its own requests further, on the wire only.

Resuming a question (deferred tool results, an interruption, a suspension) requires
the host to carry the capability state from the run being resumed, alongside the
message history. Without it the identity of the question in progress is unknowable
and the run fails rather than silently treating it as a new question.

## State

Capabilities use a plain `state: dict[str, Any]` attribute on agent dependencies when one is available. RAG state lives under `"rag"`; analysis state lives under `"analysis"`. This keeps state independent of any transport or UI protocol.

Applications serving AG-UI should adapt the agent with Pydantic AI's `AGUIAdapter`. Native model and tool events require no haiku.rag-specific bridge.

## Database path

Both factories resolve their database in this order:

1. The `db_path` argument.
2. `HAIKU_RAG_DB`.
3. `config.storage.data_dir / "haiku.rag.lancedb"`.
