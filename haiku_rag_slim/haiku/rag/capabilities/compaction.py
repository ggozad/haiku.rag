from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext

from haiku.rag.capabilities._base import RAGCapabilityBase
from haiku.rag.capabilities.evidence import (
    DiscoveredEvidence,
    discover_evidence,
    question_in_progress,
)
from haiku.rag.store.models.citation import Citation
from haiku.rag.tools.search import RETRIEVED_IMAGE_TAG, decode_picture

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


def picture_label(chunk_id: str, self_ref: str, collection: str | None = None) -> str:
    named = f"Collection: {collection}. " if collection else ""
    return (
        f"Page image retrieved from the knowledge base for cited evidence "
        f"[{chunk_id}] ({self_ref}). {named}"
        f"Not provided by the user. {RETRIEVED_IMAGE_TAG}"
    )


@dataclass(frozen=True)
class RetainedPicture:
    """A picture to re-attach, with the label that must accompany it.

    Addressed by owner, collection, document and reference, because a reference
    such as ``#/pictures/0`` repeats across all three. The label travels with it
    so it can never be emitted without its image.
    """

    capability: str
    chunk_id: str
    document_id: str
    self_ref: str
    label: str
    source: str | None = None


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

    def render(self, *, include_collection: bool = False) -> str:
        """Render an entry, optionally naming its collection."""
        title = self.citation.document_title
        uri = self.citation.document_uri
        document = f'"{title}"' if title else uri
        if title and uri and uri != title:
            document = f"{document} ({uri})"
        if include_collection and self.citation.source:
            header = (
                f"[{self.chunk_id}]\nCollection: {self.citation.source}\n"
                f"Source: {document}"
            )
        else:
            header = f"[{self.chunk_id}] Source: {document}"
        return f"{header}\n{self.citation.content}"


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
    seen: set[tuple[str, str | None, str, str]] = set()
    # A capsule may combine citations from different search scopes.
    include_collection = len({entry.citation.source for entry in entries}) > 1
    position = 0
    current_question: int | None = None
    for entry in entries:
        if entry.question != current_question:
            position += 1
            current_question = entry.question
            lines.append(group_label(position))
        lines.append(entry.render(include_collection=include_collection))
        for self_ref in entry.citation.picture_refs:
            # Overlapping chunks cite one figure, and a provider counts it twice.
            # A reference such as `#/pictures/0` repeats across documents, and a
            # document repeats across collections.
            identity = (
                entry.capability,
                entry.citation.source,
                entry.citation.document_id,
                self_ref,
            )
            if identity in seen:
                continue
            seen.add(identity)
            pictures.append(
                RetainedPicture(
                    capability=entry.capability,
                    chunk_id=entry.chunk_id,
                    document_id=entry.citation.document_id,
                    self_ref=self_ref,
                    label=picture_label(
                        entry.chunk_id,
                        self_ref,
                        entry.citation.source if include_collection else None,
                    ),
                    source=entry.citation.source,
                )
            )
    return Capsule(text=ENTRY_SEPARATOR.join(lines), pictures=tuple(pictures))


def _strip_our_pictures(part: UserPromptPart) -> UserPromptPart | None:
    """Drop the pictures we attached, together with the labels describing them.

    Ours is a label carrying the machine tag immediately followed by an image —
    both halves required. Position alone is not ownership, since several tools'
    results can arrive in one request; prose alone is not either, because a user can
    write any phrase, and treating one as proof removed a user's own picture along
    with their text. A label is only ever dropped with its picture: left behind it
    would tell the model a figure is present when it is gone.
    """
    if isinstance(part.content, str):
        return part
    items = list(part.content)
    kept: list[Any] = []
    index = 0
    while index < len(items):
        item = items[index]
        following = items[index + 1] if index + 1 < len(items) else None
        is_ours = (
            isinstance(item, str)
            and RETRIEVED_IMAGE_TAG in item
            and isinstance(following, BinaryContent)
        )
        if is_ours:
            index += 2
            continue
        kept.append(item)
        index += 1
    return replace(part, content=kept) if kept else None


def compact_history(
    messages: list[ModelMessage],
    *,
    boundary: int,
    owned_tools: frozenset[str],
    capsule_text: str,
    capsule_images: Sequence[str | BinaryContent] = (),
) -> list[ModelMessage]:
    """Replace earlier questions' evidence with the capsule, on a copy.

    ``boundary`` is how many messages existed when the current question arrived, so
    everything below it belongs to an earlier one. It comes from the recorded
    question identity rather than from message shape: mid-question a user-role part
    is as likely to be page images or an injected notice, and reading either as the
    next question strips evidence the model is still answering from.

    The newest earlier return carries the capsule and every other becomes a receipt,
    so exactly one capsule exists by construction. Returns are never removed, only
    rewritten, which keeps each one paired with its call. Nothing outside this
    capability's evidence tools is touched — not a cite acknowledgement, not another
    capability's output, not a picture the user attached.
    """
    if boundary <= 0:
        return messages

    carrier = _newest_owned_return(messages, boundary, owned_tools)
    compacted = list(messages)
    for index, message in enumerate(messages[:boundary]):
        if not isinstance(message, ModelRequest):
            continue
        parts: list[Any] = []
        for position, part in enumerate(message.parts):
            if isinstance(part, ToolReturnPart) and part.tool_name in owned_tools:
                carries = (index, position) == carrier
                body = capsule_text or RECEIPT if carries else RECEIPT
                parts.append(replace(part, content=body))
            elif isinstance(part, UserPromptPart):
                if (kept := _strip_our_pictures(part)) is not None:
                    parts.append(kept)
            else:
                parts.append(part)
        if carrier is not None and index == carrier[0] and capsule_images:
            parts.append(UserPromptPart(content=list(capsule_images)))
        if not parts:
            # A request with no parts is not a message; whatever emptied it was not
            # ours to remove after all.
            continue
        if parts != message.parts:
            compacted[index] = replace(message, parts=parts)
    return compacted


def _require_a_record_of_what_was_cited(
    evidence: Sequence[DiscoveredEvidence],
    messages: list[ModelMessage],
    boundary: int,
) -> None:
    """Refuse to compact a capability's evidence when its record was not carried.

    Judged per capability, and only for one whose own evidence is actually at
    stake: another capability's carried record says nothing about this one's, and a
    capability the model never used has nothing to lose. Without the record there
    is no capsule to put in the evidence's place, so compacting would drop it and
    leave the citations the host already displayed as the only trace.
    """
    for found in evidence:
        if found.state_carried:
            continue
        if _newest_owned_return(messages, boundary, found.tool_names) is None:
            continue
        raise RuntimeError(
            f"Evidence compaction found {found.capability} evidence from an earlier "
            "question but no record of what it cited, so replacing it would retain "
            "nothing. The host must carry the capability state between runs, "
            f"alongside the message history: {found.capability} state was missing."
        )


def _newest_owned_return(
    messages: list[ModelMessage], boundary: int, owned_tools: frozenset[str]
) -> tuple[int, int] | None:
    """Where the last of our evidence returns is, as message and part.

    The part matters: a model can call search twice in one response, so one request
    can hold several of our returns, and giving the capsule to each duplicates the
    whole of it.
    """
    for index in range(min(boundary, len(messages)) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ModelRequest):
            continue
        for position in range(len(message.parts) - 1, -1, -1):
            part = message.parts[position]
            if isinstance(part, ToolReturnPart) and part.tool_name in owned_tools:
                return index, position
    return None


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

    built_for: tuple[str | None, int] | None = field(default=None, repr=False)
    capsule: Capsule = field(default_factory=Capsule, repr=False)
    images: tuple[str | BinaryContent, ...] = field(default=(), repr=False)

    @classmethod
    def from_spec(cls) -> "EvidenceCompactionCapability":
        """Build from an agent spec. The factory takes no configuration, so
        neither does the spec surface."""
        return create_capability()

    async def for_run(self, ctx: RunContext[Any]) -> "EvidenceCompactionCapability":
        """Give the run its own build cache, so concurrent runs cannot share one."""
        return replace(self, built_for=None, capsule=Capsule(), images=())

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Rewrite the request, never the stored history.

        Deliberately not ``before_model_request``: that hook's result is assigned
        back onto the run's message history, which would destroy the host's record
        of what was retrieved and break the message counts that question identities
        and epochs are derived from.
        """
        evidence = discover_evidence(ctx)
        boundary = question_in_progress(evidence)
        owned_tools = frozenset().union(*(found.tool_names for found in evidence))
        if boundary > 0:
            _require_a_record_of_what_was_cited(
                evidence, request_context.messages, boundary
            )
        if boundary > 0:
            await self._build_once(ctx, evidence)
            request_context.messages = compact_history(
                request_context.messages,
                boundary=boundary,
                owned_tools=owned_tools,
                capsule_text=self.capsule.text,
                capsule_images=self.images,
            )
        return await handler(request_context)

    async def _build_once(
        self, ctx: RunContext[Any], evidence: Sequence[DiscoveredEvidence]
    ) -> None:
        """Build the capsule once per model request, however often the hook runs.

        Keyed on the run and its step rather than persisted: a stored key would
        freeze one question's capsule across the next.
        """
        key = (ctx.run_id, ctx.run_step)
        if key == self.built_for:
            return
        self.capsule = build_capsule(evidence)
        self.images = await self._rehydrate(ctx)
        self.built_for = key

    async def _rehydrate(self, ctx: RunContext[Any]) -> tuple[str | BinaryContent, ...]:
        """Fetch the cited pictures through the capability that retrieved them.

        Bytes are never stored in state, and the owner already holds an open
        connection. A picture that cannot be fetched, for any reason, or that will
        not decode, is emitted with neither its image nor its label: a label can
        never outlive what it describes, and a figure the model has already been
        given in text is not worth failing a question over.
        """
        owners = {
            capability.state_namespace: capability
            for capability in ctx.capabilities.values()
            if isinstance(capability, RAGCapabilityBase)
        }
        content: list[str | BinaryContent] = []
        for retained in self.capsule.pictures:
            # Indexed, not looked up defensively: the capsule was built from these
            # same capabilities in this same call, so a missing owner is a broken
            # invariant rather than a picture to skip.
            owner = owners[retained.capability]
            try:
                data = await owner.get_picture_bytes(
                    retained.document_id, retained.self_ref, retained.source
                )
            except Exception:
                # A read that fails costs this picture, not the answer.
                continue
            if data is None:
                continue
            picture = decode_picture(data, retained.self_ref)
            if picture is None:
                continue
            content.append(retained.label)
            content.append(picture)
        return tuple(content)


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
    "EvidenceCompactionCapability",
    "RetainedPicture",
    "build_capsule",
    "compact_history",
    "create_capability",
    "group_label",
    "picture_label",
]
