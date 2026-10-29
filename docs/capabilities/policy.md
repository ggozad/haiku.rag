# Citation policy capability

`CitationPolicyCapability` requires every answer to declare what grounds it. The RAG
capability's instructions ask the model to cite and record what it cites, but nothing
enforces the declaration without this capability.

Register it alongside the RAG capability, with the host carrying capability state
between runs, as in [Compose an agent](index.md#compose-an-agent). Without that state
a follow-up about evidence cited earlier goes unenforced. An agent takes at most one
policy capability. Pydantic AI rejects a second.

## Declaring nothing is a valid answer

A model that finds nothing relevant calls the cite tool with an empty list. That records
the answer as *ungrounded*, which is distinct from an answer that declared nothing at
all (*missing*). A model can always satisfy the policy without inventing grounding.

## What happens when a question ends undeclared

The model is asked once to record what grounded the answer it already gave. It is not
asked to change the answer. If the cite tool is no longer available by then, or the
question finishes undeclared anyway, the question is recorded in
`CitationPolicyState.violations` under the `"citation_policy"` state key, and the model
is not asked again.

### Structured answers

For structured output, the reminder is an output-validation retry after successful
schema validation, not a queued user message. It asks the model to register its
citations and resubmit the answer in the required output format. Invalid schema
candidates and partial streaming output do not trigger citation reminders.

The reminder consumes the host agent's output-retry budget. For example,
`retries={"output": 2}` allows one schema-validation retry and one citation retry.
Exhausting that budget raises Pydantic AI's retry-limit error. An answer that ignores
the citation reminder and still finishes undeclared is recorded as a violation,
as for plain-text answers.

## Which answers are enforced

Every answer in a conversation that has something to declare: either this question
retrieved evidence, or the conversation has already cited something, which stays
available to later answers. A follow-up about evidence cited earlier is enforced even
though it searched nothing.

Once anything has been cited, later turns are enforced too, a greeting included. The
model satisfies the policy by citing an empty list, at the cost of one extra request. A
conversation with neither a current-question evidence outcome nor any earlier citation
is not enforced.
