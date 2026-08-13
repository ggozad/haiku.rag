# Citation policy capability

`CitationPolicyCapability` requires every answer to declare what grounds it. Citing is
always available and always recorded without it, but nothing makes the model do it.

Register it alongside an evidence capability:

```python
from pydantic_ai import Agent
from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import create_capability as rag

agent = Agent(
    "openai:gpt-5",
    capabilities=[rag(db_path="my.lancedb"), citation_policy()],
)
```

It exposes no tools and takes no configuration. Exactly one policy capability makes the
decision, however many evidence capabilities are registered, so two of them cannot each
demand a citation for one answer.

## Declaring nothing is a valid answer

A model that finds nothing relevant calls the cite tool with an empty list. That records
the answer as *ungrounded*, which is distinct from an answer that declared nothing at
all (*missing*). The distinction is what makes a declaration requirable without forcing
the model to invent grounding.

## What happens when a question ends undeclared

The model is asked once to record what grounded the answer it already gave. It is not
asked to change the answer. If the cite tool is no longer available by then, or the
question finishes undeclared anyway, the question is recorded in
`CitationPolicyState.violations` under the `"citation_policy"` state key. Pointing a
model at a tool that is gone costs it retries, so the capability records the failure
instead.

## Which answers are enforced

Every answer in a conversation that has something to declare: either this question
retrieved evidence, or the conversation has already cited something, which stays
available to later answers. A follow-up about evidence cited earlier is enforced even
though it searched nothing, which is the case the capability exists for.

Once anything has been cited, later turns are enforced too, a greeting included. The
model satisfies the policy by citing an empty list, at the cost of one extra request. A
conversation with neither a current-question evidence outcome nor any earlier citation
is not enforced.
