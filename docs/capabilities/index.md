# Capabilities

haiku.rag provides native [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/):

| Capability | Use it for |
|---|---|
| [`RAGCapability`](rag.md) | Grounded document search and citations. |
| [`AnalysisCapability`](analysis.md) | Corpus computation and structural analysis with sandboxed Python. |
| [`EvidenceCompactionCapability`](compaction.md) | Optional. Shrinking a conversation's history to the evidence that was cited. |
| [`CitationPolicyCapability`](policy.md) | Optional. Requiring every answer to declare what grounds it. |

The two evidence capabilities are deferred by default. An agent initially sees only their descriptions and the standard `load_capability` tool. Instructions and tools enter the model context only when the model loads a capability.

## Compose an agent

Pick one evidence capability, and add both optional capabilities to it:

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[
        rag(db_path="my.lancedb"),
        compaction(),
        citation_policy(),
    ],
)

result = await agent.run("What does the knowledge base say about X?")
print(result.output)
```

Swap `rag` for `analysis` for an analysis agent. Both optional capabilities work the
same way with either one, and neither exposes tools or takes configuration.

!!! note "Register one evidence capability, not both"

    `RAGCapability` and `AnalysisCapability` overlap. Both search the same corpus and
    both register citations, so an agent holding both must choose between two
    near-identical search tools, and its citations land in whichever capability it
    happened to call. Each also carries its own request limit and its own search
    budget, so registering both doubles what a question may spend.

    Choose by what the questions need. `RAGCapability` answers questions from retrieved
    passages. `AnalysisCapability` adds a Python sandbox and a document filesystem, for
    questions that compute over many documents or read their structure, and it can
    search too. If you need computation, register the analysis capability alone rather
    than adding it to the RAG one.

## State

Capabilities use a plain `state: dict[str, Any]` attribute on agent dependencies when one is available. RAG state lives under `"rag"`; analysis state lives under `"analysis"`. This keeps state independent of any transport or UI protocol.

Applications serving AG-UI should adapt the agent with Pydantic AI's `AGUIAdapter`. Native model and tool events require no haiku.rag-specific bridge.

## Database path

Both factories resolve their database in this order:

1. The `db_path` argument.
2. `HAIKU_RAG_DB`.
3. `config.storage.data_dir / "haiku.rag.lancedb"`.
