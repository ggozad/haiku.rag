from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

CitationStatus = Literal["missing", "grounded", "ungrounded"]


class EvidenceRef(BaseModel):
    """One piece of evidence, identified by its owner as well as its chunk.

    A chunk id alone is not an identity: the same id can be reported by more than
    one capability, and ownership is what tells compaction whose output it may
    touch.
    """

    capability: str
    chunk_id: str


class EvidenceOccurrence(BaseModel):
    """Which questions retrieved a piece of evidence, and which cited it."""

    capability: str
    chunk_id: str
    retrieved_in_questions: list[int] = Field(default_factory=list)
    cited_in_questions: list[int] = Field(default_factory=list)


class CitationDeclaration(BaseModel):
    """What a question declared as its grounding, and when.

    Bound to a question *and* an epoch: the epoch outlives a question, so matching
    it alone would let a question that gathered no evidence inherit the previous
    declaration and read as compliant having declared nothing.
    """

    question: int
    epoch: int
    refs: list[EvidenceRef] = Field(default_factory=list)


class CapabilityEvidenceRecord(BaseModel):
    """What one evidence capability wrote, in its own state namespace.

    Holds no content: ``Citation`` in ``citation_index`` is the canonical record
    and already persists content, document id and picture refs. This is the index
    over it that compaction needs and the transcript cannot provide.

    Single-writer by construction. A record shared between capabilities would be
    overwritten by whichever of them synced its state last; merging happens in the
    transient views built by ``citation_status`` and the optional capabilities.

    ``question`` is the number of messages that existed when the question arrived,
    and ``epoch`` the number when an outcome occurred. Both are derived from the
    conversation rather than counted locally, so every participant computes the
    same values without sharing a counter. ``question`` is unset until a run
    establishes it, so a record a host merely created is distinguishable from one
    that has been through a question.
    """

    occurrences: dict[str, EvidenceOccurrence] = Field(default_factory=dict)
    question: int | None = None
    latest_evidence_epoch: int = 0
    declaration: CitationDeclaration | None = None

    def _reject_regression(self, count: int, what: str) -> None:
        """Refuse a message count below one already recorded.

        Identities and epochs are both message counts, and every comparison
        between them assumes the conversation only grows. One capability
        truncating or reordering the history breaks that, and each way of
        recording it has to refuse the same way: an unchecked evidence outcome
        freezes every later declaration as stale, while an unchecked declaration
        replaces a newer one with an older one and revives the answer it grounded.
        """
        recorded = max(
            self.question or 0,
            self.latest_evidence_epoch,
            self.declaration.epoch if self.declaration else 0,
        )
        if count < recorded:
            raise ValueError(
                f"{what} at message count {count} is behind {recorded}, which is "
                "already recorded: message history must be append-only for "
                "question identities and epochs to hold."
            )

    def begin_question(self, identity: int) -> None:
        """Take the identity of a question that has just arrived."""
        self._reject_regression(identity, "A question")
        self.question = identity

    def note_evidence(self, epoch: int) -> None:
        """Record that the model has seen an evidence outcome.

        Called for anything an answer could rest on, including a search that
        returned nothing and a failed execution that still printed output — a
        fruitless search grounds a refusal. Not called for a failure that yields
        no evidence at all, such as an exhausted budget.
        """
        self._reject_regression(epoch, "Evidence")
        self.latest_evidence_epoch = epoch

    def declare(
        self,
        refs: list[EvidenceRef],
        *,
        epoch: int,
        retrieved_now: set[str] | None = None,
    ) -> None:
        """Record validated citations for the current question.

        Repeated calls at the same epoch merge, so citing again cannot narrow what
        was already declared: an empty call after a grounded one leaves it
        grounded. A call at a later epoch declares afresh, because evidence the
        model saw in between may be what it is now citing.
        """
        if self.question is None:
            raise ValueError(
                "Citations cannot be declared before a run establishes the "
                "question identity."
            )
        self._reject_regression(epoch, "A declaration")
        current = self.declaration
        if current is not None and (current.question, current.epoch) == (
            self.question,
            epoch,
        ):
            known = {(ref.capability, ref.chunk_id) for ref in current.refs}
            current.refs.extend(
                ref for ref in refs if (ref.capability, ref.chunk_id) not in known
            )
        else:
            self.declaration = CitationDeclaration(
                question=self.question, epoch=epoch, refs=list(refs)
            )

        for ref in refs:
            occurrence = self.occurrences.setdefault(
                ref.chunk_id,
                EvidenceOccurrence(capability=ref.capability, chunk_id=ref.chunk_id),
            )
            if self.question not in occurrence.cited_in_questions:
                occurrence.cited_in_questions.append(self.question)
            if (
                retrieved_now
                and ref.chunk_id in retrieved_now
                and self.question not in occurrence.retrieved_in_questions
            ):
                occurrence.retrieved_in_questions.append(self.question)


def citation_status(
    records: Iterable[CapabilityEvidenceRecord], *, question: int
) -> CitationStatus:
    """Derived, never stored, so refs and status cannot contradict.

    A declaration is current only for the question it was made in, and only if it
    followed the newest evidence outcome of *every* capability: a question where
    one capability cited and another then searched without citing is not grounded.
    Strictly later, since a citation made in the same request as an outcome cannot
    have read it.

    A grounding *violation* is not one of these: that is an enforcement outcome
    recorded by the policy capability, not something the model declared.
    """
    records = list(records)
    horizon = max((record.latest_evidence_epoch for record in records), default=0)
    current = [
        record.declaration
        for record in records
        if record.declaration is not None
        and record.declaration.question == question
        and record.declaration.epoch > horizon
    ]
    if not current:
        return "missing"
    return (
        "grounded" if any(declaration.refs for declaration in current) else "ungrounded"
    )


__all__ = [
    "CapabilityEvidenceRecord",
    "CitationDeclaration",
    "CitationStatus",
    "EvidenceOccurrence",
    "EvidenceRef",
    "citation_status",
]
