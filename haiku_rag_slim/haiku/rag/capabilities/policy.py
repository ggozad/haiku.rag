from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext

from haiku.rag.capabilities.evidence import (
    DiscoveredEvidence,
    discover_evidence,
    question_in_progress,
)
from haiku.rag.capabilities.ledger import citation_status

CAPABILITY_ID = "haiku-rag-citation-policy"

STATE_NAMESPACE = "citation_policy"

REDIRECT_HINT = "record what grounded the answer you already gave"

CITATION_REDIRECT_TAG = "[haiku.rag/citation-redirect]"
"""Tag the redirect carries, so a question can tell it has already been asked.

Not the wording: a user writing "record what grounded the answer you already gave"
in their own question would otherwise read as a redirect we had sent, silently
switching enforcement off for that question.
"""

REDIRECT = (
    "You answered without registering citations. This asks you to "
    f"{REDIRECT_HINT} — it is not a request to change that answer, and not a "
    "signal that it was wrong. Call the cite tool with the chunk_ids that support "
    "it. If nothing in the knowledge base supports it, or you said you could not "
    "find the information, call it with an empty list. Then repeat your answer "
    f"exactly as you gave it. {CITATION_REDIRECT_TAG}"
)


class CitationPolicyState(BaseModel):
    """What the policy decided, for hosts and evaluations to read.

    ``violations`` holds the identities of questions that ended undeclared while
    the cite tool was already gone, so no redirect was possible. It is an
    enforcement outcome, which is why it lives here rather than in an evidence
    capability's record: nothing the model declared says it.
    """

    violations: list[int] = Field(default_factory=list)


@dataclass
class CitationPolicyCapability(AbstractCapability[Any]):
    """Requires every answer to declare what grounds it, once per question.

    Registering it is the only switch. Without it citations are still recorded and
    still validated, they are simply not required.

    Enforcement needs exactly one decision-maker. If each evidence capability
    enforced its own citations, both could redirect the model within one question
    and neither could see what the other had declared, so this capability
    discovers them all and merges their records before deciding.

    Registering two is rejected by pydantic-ai before the run starts, since they
    would share this capability's id.
    """

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Decide once, at the last moment a question can still be redirected.

        A response carrying no tool calls ends the question, so there is no later
        opportunity. Citing is unconditional, so an undeclared answer is a protocol
        breach whether the model answered or refused, and this never has to guess
        which it was.
        """
        if any(isinstance(part, ToolCallPart) for part in response.parts):
            return response
        evidence = discover_evidence(ctx)
        question = question_in_progress(evidence)
        if not _has_evidence_to_declare(evidence):
            return response
        records = [found.record for found in evidence]
        if citation_status(records, question=question) != "missing":
            return response
        if _already_asked(ctx.messages, question):
            return response

        if any(found.cite_available for found in evidence):
            ctx.enqueue(REDIRECT, priority="when_idle")
        else:
            self._record_violation(ctx, question)
        return response

    def _record_violation(self, ctx: RunContext[Any], question: int) -> None:
        """Note a question that could not be asked to cite, the tool being gone.

        Recorded once per question: a resumption of the same question decides
        again, and one question is one outcome.
        """
        outer = getattr(ctx.deps, "state", None)
        if not isinstance(outer, dict):
            return
        state = CitationPolicyState.model_validate(outer.get(STATE_NAMESPACE) or {})
        if question not in state.violations:
            state.violations.append(question)
        outer[STATE_NAMESPACE] = state.model_dump(mode="json")

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Publish an empty outcome, so a host can tell "none" from "not running"."""
        outer = getattr(ctx.deps, "state", None)
        if isinstance(outer, dict):
            outer.setdefault(
                STATE_NAMESPACE, CitationPolicyState().model_dump(mode="json")
            )


def _already_asked(messages: list[ModelMessage], question: int) -> bool:
    """Whether this question has already been asked to declare its grounding.

    Read from the history rather than remembered on the instance, which a
    resumption's ``for_run`` would forget — the same question would then be asked
    twice. It also makes the right call when a redirect was enqueued but the run
    ended before it reached the model: nothing is in the history, so it is asked
    again, which is what the model needs.

    Matched on the machine tag rather than the wording, so a question that happens
    to contain the phrase cannot pass as a redirect we sent.
    """
    return any(
        isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and CITATION_REDIRECT_TAG in part.content
        for message in messages[question:]
        for part in message.parts
    )


def _has_evidence_to_declare(evidence: list[DiscoveredEvidence]) -> bool:
    """Whether anything exists that this answer could have been grounded on.

    Either this question produced an evidence outcome, or the conversation has
    already cited something — which stays available to a later answer, in a capsule
    if a compactor is registered and in full if not. Requiring a fresh outcome
    exempted exactly the follow-up that reuses earlier evidence, which is the case
    enforcement exists for.

    A conversation that has neither has nothing to declare: a greeting, an aside.
    Read from the ledger rather than from ``state.searches``, which a new question
    clears, so an answer grounded on code execution or a document read counts too.
    """
    question = question_in_progress(evidence)
    return any(
        found.record.latest_evidence_epoch > question or found.record.occurrences
        for found in evidence
    )


def create_capability() -> CitationPolicyCapability:
    """Create the capability that requires an answer to declare its grounding."""
    return CitationPolicyCapability(
        id=CAPABILITY_ID,
        description=(
            "Requires every answer to register the evidence that grounds it, or to "
            "declare that nothing does."
        ),
    )


__all__ = [
    "CAPABILITY_ID",
    "CITATION_REDIRECT_TAG",
    "REDIRECT",
    "REDIRECT_HINT",
    "STATE_NAMESPACE",
    "CitationPolicyCapability",
    "CitationPolicyState",
    "create_capability",
]
