# Evidence compaction capability

`EvidenceCompactionCapability` keeps a multi-turn conversation from carrying every
search result it ever produced. Every question adds its evidence to the history, so
requests grow turn after turn, which degrades answers and can exceed a provider's
limits.

Register it alongside an evidence capability:

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[rag(db_path="my.lancedb"), compaction()],
)
```

It exposes no tools and takes no configuration. Registering it is the only switch:
leave it out and the transcript reaches the model untouched.

## What it does

On each request, evidence from earlier questions is replaced by the evidence those
questions actually cited. Cited text and cited page images are kept in full, grouped by
the question that cited them, and stay citable by the same chunk ids. Every other
earlier evidence return becomes a short receipt. The current question is untouched.

Compaction rewrites the request, never the stored history, so `all_messages()` still
holds everything the run gathered.

This reduces what a request carries. It does not bound it: retained evidence still
grows with the conversation. A host that needs more aggressive pruning can compact its
own requests further, on the wire only.

## Resuming a question

Resuming a question (deferred tool results, an interruption, a suspension) requires the
host to carry the capability state from the run being resumed, alongside the message
history. Without it the identity of the question in progress is unknowable, and the run
fails rather than silently treating it as a new question.
