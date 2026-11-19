import base64
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel
from pydantic_ai import ToolFailed
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from haiku.rag.client import HaikuRAG
from haiku.rag.client.search import expand_near
from haiku.rag.store.models.chunk import (
    Content,
    SearchResult,
    TimeWindow,
    qualified_id,
)
from haiku.rag.store.models.document import Recording
from haiku.rag.store.models.document_item import (
    DocumentItem,
    merge_time_ranges,
    spread_frames,
)
from haiku.rag.utils.formatting import format_time_range
from haiku.rag.utils.images import CITED_EVIDENCE, RETRIEVED_IMAGE_TAG, picture_keys

VIEW_MAX_FRAMES = 8

IMAGES_OMITTED = (
    "[Page images retrieved from the knowledge base are omitted here: the request "
    "reached its image limit.]"
)


class CodeExecutionEntry(BaseModel):
    code: str
    stdout: str
    stderr: str = ""
    success: bool = True
    search_calls: int = 0


EvidenceKey = tuple[tuple[str | None, str | None], tuple[str, frozenset]]
"""What tells one rendered result from another: qualified id, then signature.

The qualified id comes first because the rendered string alone would conflate
identical renderings of the same chunk id held by two databases.
"""


def evidence_signature(result: SearchResult) -> tuple:
    """The rendered evidence a result shows the model, as an equivalence key.

    Rank and total are held at neutral values, and the collection label is
    left out: position, score and presentation must not tell two renderings
    of one chunk apart. The qualified id keeps copies from different
    collections distinct.
    """
    return (result.format_for_agent(rank=0, total=0), picture_keys(result))


def evidence_key(result: SearchResult) -> EvidenceKey:
    return (qualified_id(result.source, result.chunk_id), evidence_signature(result))


def require_collections(requested: Sequence[str], available: Sequence[str]) -> None:
    """Fail the call on a name outside the collections this run covers."""
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ToolFailed(
            f"Unknown collection(s): {', '.join(unknown)}. "
            f"This run covers: {', '.join(available)}."
        )


def without_sources(tool_def: ToolDefinition) -> ToolDefinition:
    """The tool's definition without its `sources` or `source` parameter."""
    schema = dict(tool_def.parameters_json_schema)
    schema["properties"] = {
        name: spec
        for name, spec in schema.get("properties", {}).items()
        if name not in ("sources", "source")
    }
    if "required" in schema:
        schema["required"] = [
            name for name in schema["required"] if name not in ("sources", "source")
        ]
    return replace(tool_def, parameters_json_schema=schema)


async def search_corpus(
    rag: HaikuRAG,
    query: str,
    limit: int | None = None,
    document_filter: str | None = None,
    sources: list[str] | None = None,
    shown: AbstractSet[EvidenceKey] = frozenset(),
    *,
    include_collection: bool,
    content: Content | None = None,
    window: TimeWindow | None = None,
    include_images: bool = True,
) -> tuple[str, list[SearchResult], set[EvidenceKey]]:
    """Search and context-expand results, eliding evidence already shown.

    Returns the formatted results, the full result list and the evidence keys
    the formatting rendered in full. ``include_collection`` names each result's
    collection, a decision about the run that the caller makes. A result whose
    key is in ``shown`` keeps its slot but collapses to one line; the result
    list is never filtered. `content`, `window` and `include_images` pass to
    `HaikuRAG.search`, and `window` to the expansion.
    """
    results = await rag.search(
        query,
        limit=limit,
        filter=document_filter,
        include_images=include_images,
        sources=sources,
        content=content,
        window=window,
    )
    if content in ("speech", "frames"):
        results = await expand_near(rag, results, window)
    else:
        results = await rag.expand_context(results, window=window)
    rendered: set[EvidenceKey] = set()
    parts: list[str] = []
    total = len(results)
    for index, result in enumerate(results):
        key = evidence_key(result)
        if key in shown or key in rendered:
            parts.append(
                f"Also matched, shown above: [{result.chunk_id}] "
                f"[rank {index + 1} of {total}]"
            )
        else:
            parts.append(
                result.format_for_agent(
                    rank=index + 1, total=total, include_collection=include_collection
                )
            )
            rendered.add(key)
    formatted = "\n\n---\n\n".join(parts)
    return formatted or "No results found.", list(results), rendered


@dataclass
class RecordingView:
    """What `view` returned: a result per speech chunk holding only the lines
    shown, and one per frame chunk holding only its frame, in time order."""

    results: list[SearchResult]
    continues_at: float | None


async def view_recording(
    rag: HaikuRAG,
    recording: Recording,
    window: TimeWindow,
    max_chars: int,
    include_images: bool,
) -> RecordingView:
    """The sentences of `recording` inside `window`, whole, up to `max_chars`
    (at least one), and its frames, spread across the window down to
    `VIEW_MAX_FRAMES` when it holds more."""
    items = await rag.document_item_repository.get_items_in_window(window)
    lines = [
        item
        for item in items
        if item.label != "picture" and item.text and item.start_time is not None
    ]
    shown: list[DocumentItem] = []
    used = 0
    continues_at = None
    for index, item in enumerate(lines):
        # The line adds `[HH:MM:SS] ` before the text and a newline after it.
        cost = len(item.text) + 12
        if shown and used + cost > max_chars:
            # Past the end of everything shown, so reading on from here skips
            # every item already returned, an overlapping one included. Items
            # ending before it are never read on, so they stay on this page.
            continues_at = max(i.end_time for i in shown if i.end_time is not None)
            shown.extend(
                rest
                for rest in lines[index:]
                if rest.end_time is not None and rest.end_time <= continues_at
            )
            break
        shown.append(item)
        used += cost
    frames = spread_frames(
        {
            item.self_ref: item.start_time
            for item in items
            if item.label == "picture" and item.start_time is not None
        },
        VIEW_MAX_FRAMES,
    )
    index = (
        await rag.chunk_repository.get_chunk_ids_by_self_ref_grouped(
            [recording.document_id]
        )
    ).get(recording.document_id, {})
    # An item two chunks share is shown once, under the first of them.
    members: dict[str, list[DocumentItem]] = {}
    for item in shown:
        for chunk_id in index.get(item.self_ref, [])[:1]:
            members.setdefault(chunk_id, []).append(item)
    frame_items = {item.self_ref: item for item in items if item.self_ref in frames}
    for ref in frames:
        for chunk_id in index.get(ref, []):
            members.setdefault(chunk_id, []).append(frame_items[ref])

    results = []
    for chunk_id, chunk_items in members.items():
        chunk = await rag.chunk_repository.get_by_id(chunk_id)
        if chunk is None:  # pragma: no cover - the index was read from the table
            continue
        result = SearchResult.from_chunk(chunk, score=1.0)
        result.source = recording.source
        result.document_title = recording.title
        result.document_uri = recording.uri
        lines, pictures, spans = [], {}, []
        for item in chunk_items:
            assert item.start_time is not None and item.end_time is not None
            spans.append((item.start_time, item.end_time))
            clock = format_time_range((item.start_time, item.start_time))
            if item.label == "picture":
                lines.append(f"[{clock}] (frame)")
                pictures[item.self_ref] = item.start_time
            else:
                lines.append(f"[{clock}] {item.text}")
        result.content = "\n".join(lines)
        result.doc_item_refs = [item.self_ref for item in chunk_items]
        result.time_range = merge_time_ranges(spans)
        result.picture_times = pictures
        result.image_data = None
        if include_images and pictures:
            result.image_data = {}
            for ref in pictures:
                data = await rag.document_item_repository.get_picture_bytes(
                    recording.document_id, ref
                )
                if data:  # pragma: no branch - every stored frame has its bytes
                    result.image_data[ref] = base64.b64encode(data).decode("ascii")
        results.append(result)
    results.sort(
        key=lambda r: (r.time_range[0] if r.time_range else 0.0, not r.picture_times)
    )
    return RecordingView(results, continues_at)


def merge_results(
    existing: list[SearchResult], incoming: Iterable[SearchResult]
) -> None:
    """Add the results not already held.

    Identity is the database and the chunk id: results built by hand carry
    neither and cannot be told apart, so they collapse to the first.
    """
    seen = {qualified_id(result.source, result.chunk_id) for result in existing}
    for result in incoming:
        key = qualified_id(result.source, result.chunk_id)
        if key not in seen:
            existing.append(result)
            seen.add(key)


def refresh_results(
    held: Iterable[list[SearchResult]], incoming: Iterable[SearchResult]
) -> None:
    """Replace every held version of an incoming chunk with the incoming one."""
    latest = {
        qualified_id(result.source, result.chunk_id): result
        for result in incoming
        if result.chunk_id is not None
    }
    for results in held:
        for index, result in enumerate(results):
            key = qualified_id(result.source, result.chunk_id)
            if result.chunk_id is not None and key in latest:
                results[index] = latest[key]


def retrieved_pictures(content: Sequence[Any]) -> list[int]:
    """Index of each retrieved picture's label: the tag, then an image right after."""
    return [
        index
        for index, item in enumerate(content[:-1])
        if isinstance(item, str)
        and RETRIEVED_IMAGE_TAG in item
        and isinstance(content[index + 1], BinaryContent)
    ]


def _is_image(item: Any) -> bool:
    return isinstance(item, ImageUrl) or (
        isinstance(item, BinaryContent) and item.is_image
    )


def limit_retrieved_images(
    messages: list[ModelMessage], max_images: int
) -> list[ModelMessage]:
    """Drop retrieved pictures until the request holds `max_images` images.

    Every image counts, and only retrieved pictures are dropped, each with its
    label. Pictures of earlier rounds are dropped first, oldest message first,
    then pictures cited in earlier questions, then the last message's. Within a
    part the last picture is dropped first. Returns `messages` itself when
    nothing is dropped, else a copy.
    """
    total = 0
    parts: dict[tuple[int, int], UserPromptPart] = {}
    candidates: list[tuple[int, int, int, int]] = []
    newest = len(messages) - 1
    for at, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            continue
        for position, part in enumerate(message.parts):
            if isinstance(part, ToolReturnPart):
                total += sum(_is_image(item) for item in part.files)
            elif isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                total += sum(_is_image(item) for item in part.content)
                parts[at, position] = part
                for index in retrieved_pictures(part.content):
                    cited = CITED_EVIDENCE in str(part.content[index])
                    group = 2 if at == newest else 1 if cited else 0
                    candidates.append((group, at, position, -index))
    excess = total - max_images
    if excess <= 0:
        return messages
    dropped: dict[tuple[int, int], set[int]] = {}
    for _, at, position, index in sorted(candidates)[:excess]:
        dropped.setdefault((at, position), set()).update((-index, 1 - index))
    limited = list(messages)
    for (at, position), gone in dropped.items():
        part = parts[at, position]
        kept = [item for index, item in enumerate(part.content) if index not in gone]
        message = limited[at]
        rewritten = list(message.parts)
        rewritten[position] = replace(part, content=kept or [IMAGES_OMITTED])
        limited[at] = replace(message, parts=rewritten)
    return limited


__all__ = [
    "CodeExecutionEntry",
    "EvidenceKey",
    "evidence_key",
    "evidence_signature",
    "merge_results",
    "refresh_results",
    "require_collections",
    "retrieved_pictures",
    "search_corpus",
    "without_sources",
]
