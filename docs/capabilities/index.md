# Capabilities

haiku.rag provides native [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/):

| Capability | Use it for |
|---|---|
| [`RAGCapability`](rag.md) | Grounded document search, sandboxed Python over the corpus, and citations. |
| [`EvidenceCompactionCapability`](compaction.md) | Optional. Shrinking a conversation's history to the evidence that was cited. |
| [`CitationPolicyCapability`](policy.md) | Optional. Requiring every answer to declare what grounds it. |

The RAG capability is deferred by default. An agent initially sees only its description and the standard `load_capability` tool. Instructions and tools enter the model context only when the model loads it.

## Compose an agent

Add both optional capabilities to the RAG capability:

```python
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import create_capability as rag


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


agent = Agent(
    "openai:gpt-5",
    capabilities=[
        rag(db_path="my.lancedb"),
        compaction(),
        citation_policy(),
    ],
    deps_type=Deps,
)

# One Deps and one history for the conversation: the capabilities read both.
deps = Deps()
history: list[ModelMessage] = []

result = await agent.run("What does the knowledge base say about X?", deps=deps, message_history=history)
history = list(result.all_messages())
print(result.output)
```

!!! warning "Both optional capabilities need the host to carry state"

    They read what earlier questions retrieved and cited from the capability's
    state, so the host must expose a `state` dict on its agent dependencies and
    hand the same dict back on every run of a conversation, alongside the message
    history. With only the message history, every run starts from an empty record:
    compaction refuses rather than replacing evidence it cannot retain, and the
    citation policy cannot enforce a follow-up about evidence cited earlier.

Neither optional capability exposes tools or takes configuration.

## Agent specs

The capabilities can be declared in a Pydantic AI [agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml title="agent.yaml"
model: openai:gpt-5
instructions: You are a research assistant with access to a document knowledge base.
capabilities:
  - RAGCapability:
      db_path: /data/kb.lancedb
      defer_loading: false
  - EvidenceCompactionCapability
  - CitationPolicyCapability
```

Pydantic AI does not discover third-party capabilities, so the caller names the classes:

```python
from pydantic_ai import Agent

from haiku.rag.capabilities.compaction import EvidenceCompactionCapability
from haiku.rag.capabilities.policy import CitationPolicyCapability
from haiku.rag.capabilities.rag import RAGCapability

agent = Agent.from_file(
    "agent.yaml",
    deps_type=Deps,
    custom_capability_types=[
        RAGCapability,
        EvidenceCompactionCapability,
        CitationPolicyCapability,
    ],
)
```

`deps_type` stays a Python argument, since the capabilities read and write their state
through `deps.state` (see [State](#state)). `Agent.from_file` reads YAML, which needs
`pydantic-ai-slim[spec]`; `Agent.from_spec` takes a dict and needs no YAML parser.

Set `defer_loading: false` when the agent has nothing else to route to, so the tools
are visible immediately. Leave it at the default when the model should choose among
several capabilities.

A `config:` block accepts a whole `AppConfig`, for agents in one process that need
different databases or embedding models:

```yaml
capabilities:
  - RAGCapability:
      db_path: /data/kb.lancedb
      config:
        embeddings:
          model: {provider: ollama, name: embeddinggemma, vector_dim: 2048}
```

The block is read like a `haiku.rag.yaml` file: keys it omits take `AppConfig` defaults
rather than values from the configuration file on disk. The embedding model must match the
database; a mismatch may prevent opening it or produce invalid retrieval. Write the block in
full or omit it and let the [configuration file](../configuration/index.md) apply.

## State

Capabilities use a plain `state: dict[str, Any]` attribute on agent dependencies when one is available. RAG state lives under `"rag"`. This keeps state independent of any transport or UI protocol.

Applications serving AG-UI should adapt the agent with Pydantic AI's `AGUIAdapter`. Native model and tool events require no haiku.rag-specific bridge.

## Database Selection

The RAG capability covers the databases the configuration places: [`lancedb.databases`](../configuration/storage.md#multiple-databases), or with nothing configured the default database `haiku.rag` under `storage.data_dir`. The `db_path` argument places one database where the configuration places none; beside `lancedb.databases` it raises `AmbiguousDatabaseError`.

`sources` narrows that coverage to the databases it names, in a spec as in the factory:

```yaml
capabilities:
  - RAGCapability:
      sources: [manuals, specs]
```

An unknown name raises `UnknownDatabaseError`, an empty list `ValueError`. `sources` is refused beside `db_path` and beside `rag=` with `AmbiguousDatabaseError`, since each of those already says which databases the capability covers.

The `sources` field of the capability [state](#state) selects among the databases the capability covers, for one question. A question naming a database outside that coverage fails when it searches.

`document_filter` in the same state is a SQL WHERE clause over the document columns (see [Filtering Search Results](../python.md#filtering-search-results)). The host sets it, and it persists until the host changes it. It restricts what the capability's searches retrieve and which documents the sandbox mounts, not what a citation can resolve, so evidence from an earlier question stays citable after the filter narrows.

Passing a client through `rag=` bypasses this selection. The capability uses the databases covered by that client and does not close it.
