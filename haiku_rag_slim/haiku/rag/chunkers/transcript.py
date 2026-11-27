from typing import TYPE_CHECKING

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata
from haiku.rag.store.models.document_item import time_range

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, TextItem

MAX_CHUNK_SECONDS = 30.0
"""Longest stretch of speech in one chunk, unless a single segment is longer."""


def _segments(doc: "DoclingDocument") -> list["TextItem"]:
    """The recording's speech segments, in time order."""
    from docling_core.types.doc import TextItem

    return [
        item
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.text.strip()
    ]


def _transcript(text: str) -> "DoclingDocument":
    from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument

    doc = DoclingDocument(name="transcript")
    doc.add_text(label=DocItemLabel.TEXT, text=text, content_layer=ContentLayer.BODY)
    return doc


def _within_cap(
    covered: list[tuple["TextItem", tuple[int, int]]],
) -> list[list[tuple["TextItem", tuple[int, int]]]]:
    """The segments grouped in order, a group of several spanning at most
    `MAX_CHUNK_SECONDS`."""
    groups: list[list[tuple[TextItem, tuple[int, int]]]] = []
    for entry in covered:
        if groups:
            times = time_range([segment for segment, _ in [*groups[-1], entry]])
            if times is None or times[1] - times[0] <= MAX_CHUNK_SECONDS:
                groups[-1].append(entry)
                continue
        groups.append([entry])
    return groups


class TranscriptChunker(DocumentChunker):
    """Chunks a recording's transcript as one text, so the wrapped chunker splits
    between sentences, and gives each chunk the segments covered by its text. A
    chunk spanning more than `MAX_CHUNK_SECONDS` is split between segments."""

    def __init__(self, chunker: DocumentChunker):
        self.chunker = chunker

    async def chunk(self, document: "DoclingDocument | None") -> list[Chunk]:
        if document is None:
            return []
        segments = _segments(document)
        if not segments:
            return []

        joined = " ".join(s.text for s in segments)
        # The wrapped chunker rewrites whitespace, so chunks are found in the
        # transcript by their other characters.
        positions = [i for i, ch in enumerate(joined) if not ch.isspace()]
        bare = "".join(joined[i] for i in positions)
        bounds, at = [], 0
        for segment in segments:
            bounds.append((at, at + len(segment.text)))
            at += len(segment.text) + 1

        chunks = await self.chunker.chunk(_transcript(joined))
        result: list[Chunk] = []
        cursor = 0
        for chunk in chunks:
            needle = "".join(chunk.content.split())
            found = bare.find(needle, cursor)
            if found < 0:
                raise ValueError(f"Chunk not in the transcript: {chunk.content!r}")
            start, cursor = positions[found], found + len(needle)
            end = positions[cursor - 1] + 1
            covered = [
                (segment, (a, b))
                for segment, (a, b) in zip(segments, bounds)
                if a < end and start < b
            ]
            groups = _within_cap(covered)
            for group in groups:
                items = [segment for segment, _ in group]
                metadata = ChunkMetadata(
                    doc_item_refs=[i.self_ref for i in items],
                    labels=[i.label for i in items],
                ).model_dump()
                if len(groups) == 1:
                    chunk.metadata = metadata
                    result.append(chunk)
                    continue
                text = joined[max(start, group[0][1][0]) : min(end, group[-1][1][1])]
                result.append(
                    chunk.model_copy(
                        update={"content": text.strip(), "metadata": metadata}
                    )
                )
        return result
