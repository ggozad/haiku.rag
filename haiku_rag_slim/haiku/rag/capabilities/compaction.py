from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from haiku.rag.capabilities._base import RAGCapabilityBase
from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord
from haiku.rag.store.models.citation import Citation

CAPABILITY_ID = "haiku-rag-evidence-compaction"

CAPSULE_HEADER = (
    "[Evidence cited earlier in this conversation, kept so later questions can "
    "rely on it. Cite these chunk_ids directly when you use them.]"
)

RECEIPT = (
    "[Evidence retrieved for an earlier question, no longer shown. It does not "
    "count as cited for the current question.]"
)

ENTRY_SEPARATOR = "\n\n"


def group_label(position: int) -> str:
    """Name a group by its position among the groups, not by question number.

    A question identity is a message count, so a header built from it would present
    an index as a turn number, and an ordinal over the groups is not the
    conversation's ordinal either whenever a question in between cited nothing. The
    label claims only what it is: a grouping, newest first.
    """
    return f"[Cited evidence group {position}]"


def picture_label(chunk_id: str, self_ref: str) -> str:
    return (
        f"Page image retrieved from the knowledge base for cited evidence "
        f"[{chunk_id}] ({self_ref}). Not provided by the user."
    )


@dataclass(frozen=True)
class DiscoveredEvidence:
    """One evidence capability's records, as the compactor found them.

    Read-only and rebuilt per request: the compactor merges these into a view and
    persists nothing about evidence itself.
    """

    capability: str
    record: CapabilityEvidenceRecord
    citations: Mapping[str, Citation]
    tool_names: frozenset[str]


@dataclass(frozen=True)
class RetainedPicture:
    """A picture to re-attach, with the label that must accompany it.

    Addressed by owner, document and reference, because a reference such as
    ``#/pictures/0`` repeats across documents and capabilities. The label travels
    with it so it can never be emitted without its image.
    """

    capability: str
    chunk_id: str
    document_id: str
    self_ref: str
    label: str


@dataclass(frozen=True)
class Capsule:
    """Everything the compactor would insert, and nothing about where it goes."""

    text: str = ""
    pictures: tuple[RetainedPicture, ...] = ()


@dataclass(frozen=True)
class _Entry:
    capability: str
    chunk_id: str
    question: int
    citation: Citation

    def render(self) -> str:
        title = self.citation.document_title
        uri = self.citation.document_uri
        source = f'"{title}"' if title else uri
        if title and uri and uri != title:
            source = f"{source} ({uri})"
        return f"[{self.chunk_id}] Source: {source}\n{self.citation.content}"


def _eligible_entries(evidence: Sequence[DiscoveredEvidence]) -> list[_Entry]:
    """Cited evidence with content, newest citing question first.

    Evidence cited in several questions belongs to the most recent one, so it is
    rendered once and grouped where the model last used it.

    An occurrence and its canonical ``Citation`` are written by the same call, so a
    cited chunk without one is not a state this design produces. Rendering the rest
    regardless would quietly drop evidence an answer rested on, so it is reported.
    """
    entries = []
    for discovered in evidence:
        for chunk_id, occurrence in discovered.record.occurrences.items():
            if not occurrence.cited_in_questions:
                continue
            citation = discovered.citations.get(chunk_id)
            if citation is None:
                raise ValueError(
                    f"{discovered.capability} cited {chunk_id} in question(s) "
                    f"{occurrence.cited_in_questions} but has no citation record "
                    "for it, so its content cannot be retained."
                )
            entries.append(
                _Entry(
                    capability=discovered.capability,
                    chunk_id=chunk_id,
                    question=max(occurrence.cited_in_questions),
                    citation=citation,
                )
            )
    entries.sort(key=lambda entry: (-entry.question, entry.capability, entry.chunk_id))
    return entries


def build_capsule(evidence: Sequence[DiscoveredEvidence]) -> Capsule:
    """Render every cited piece of evidence, grouped by the question that cited it.

    Everything cited is kept whole and everything else is dropped. There is no
    character budget: what a model can hold is the model's business, and a knob for
    it would only half-rescue models that fail on long conversations regardless.

    A host that needs earlier evidence pruned can compact further on top, on the wire
    only. Removing or reordering the stored history breaks the message counts that
    question identities and epochs are derived from, and the next record written is
    refused.

    Pure: no I/O and no message history, so what goes on the wire stays separable
    from what it should contain. Picture bytes are fetched by the caller, which is
    why a picture travels with its label rather than beside it.
    """
    entries = _eligible_entries(evidence)
    if not entries:
        return Capsule()

    lines = [CAPSULE_HEADER]
    pictures: list[RetainedPicture] = []
    seen: set[tuple[str, str, str]] = set()
    position = 0
    current_question: int | None = None
    for entry in entries:
        if entry.question != current_question:
            position += 1
            current_question = entry.question
            lines.append(group_label(position))
        lines.append(entry.render())
        for self_ref in entry.citation.picture_refs:
            # Overlapping chunks cite one figure, and a provider counts it twice.
            # Identity is owner plus document plus reference, so the same reference
            # in another document stays a different picture.
            identity = (entry.capability, entry.citation.document_id, self_ref)
            if identity in seen:
                continue
            seen.add(identity)
            pictures.append(
                RetainedPicture(
                    capability=entry.capability,
                    chunk_id=entry.chunk_id,
                    document_id=entry.citation.document_id,
                    self_ref=self_ref,
                    label=picture_label(entry.chunk_id, self_ref),
                )
            )
    return Capsule(text=ENTRY_SEPARATOR.join(lines), pictures=tuple(pictures))


@dataclass
class EvidenceCompactionCapability(AbstractCapability[Any]):
    """Rewrites the history from what the evidence capabilities recorded.

    Registering it is what turns compaction on: a host that leaves it out gets an
    untouched transcript, which is why it has no enable flag. It reads the evidence
    capabilities through the run's registry and holds no reference to any of them,
    so a host running one capability, both, or neither needs no wiring change.

    Registering two is rejected by pydantic-ai before the run starts, since they
    would share this capability's id.
    """

    def discover(self, ctx: RunContext[Any]) -> list[DiscoveredEvidence]:
        """Read what each evidence capability recorded, without writing anything.

        The registry holds the per-run instances, which are the ones carrying
        state; the registered objects never do. That includes a deferred capability
        the model has not loaded, whose record is simply empty.
        """
        discovered = []
        for capability in ctx.capabilities.values():
            if not isinstance(capability, RAGCapabilityBase):
                continue
            state = capability.state
            discovered.append(
                DiscoveredEvidence(
                    capability=capability.state_namespace,
                    record=cast(CapabilityEvidenceRecord, cast(Any, state).evidence),
                    citations=cast(Any, state).citation_index,
                    tool_names=frozenset(capability.evidence_tool_names()),
                )
            )
        return sorted(discovered, key=lambda evidence: evidence.capability)


def create_capability() -> EvidenceCompactionCapability:
    """Create the capability that compacts history from recorded evidence."""
    return EvidenceCompactionCapability(
        id=CAPABILITY_ID,
        description=(
            "Replaces earlier questions' evidence on the wire with a capsule of "
            "what was cited."
        ),
    )


__all__ = [
    "CAPABILITY_ID",
    "CAPSULE_HEADER",
    "RECEIPT",
    "Capsule",
    "DiscoveredEvidence",
    "EvidenceCompactionCapability",
    "RetainedPicture",
    "build_capsule",
    "create_capability",
    "group_label",
    "picture_label",
]
