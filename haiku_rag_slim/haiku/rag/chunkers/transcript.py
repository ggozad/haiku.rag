from typing import TYPE_CHECKING

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, TextItem


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


class TranscriptChunker(DocumentChunker):
    """Chunks a recording's transcript as one text, so the wrapped chunker splits
    between sentences, and gives each chunk the segments covered by its text."""

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
        cursor = 0
        for chunk in chunks:
            needle = "".join(chunk.content.split())
            found = bare.find(needle, cursor)
            if found < 0:
                raise ValueError(f"Chunk not in the transcript: {chunk.content!r}")
            start, cursor = positions[found], found + len(needle)
            end = positions[cursor - 1] + 1
            items = [
                segment
                for segment, (a, b) in zip(segments, bounds)
                if a < end and start < b
            ]
            chunk.metadata = ChunkMetadata(
                doc_item_refs=[i.self_ref for i in items],
                labels=[i.label for i in items],
            ).model_dump()
        return chunks
