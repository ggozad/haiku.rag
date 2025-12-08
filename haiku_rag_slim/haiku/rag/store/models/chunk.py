from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DocItem, DoclingDocument


class BoundingBox(BaseModel):
    """Bounding box coordinates for visual grounding."""

    page_no: int
    left: float
    top: float
    right: float
    bottom: float


class ChunkMetadata(BaseModel):
    """
    Structured metadata for a chunk, including DoclingDocument references.

    Attributes:
        doc_item_refs: JSON pointer references to DocItems in the parent DoclingDocument
                       (e.g., ["#/texts/5", "#/texts/6", "#/tables/0"])
        headings: Section heading hierarchy for this chunk
                  (e.g., ["Chapter 1", "Section 1.1"])
        labels: Semantic labels for each doc_item (e.g., ["paragraph", "table"])
        page_numbers: Page numbers where the chunk content appears
    """

    doc_item_refs: list[str] = []
    headings: list[str] | None = None
    labels: list[str] = []
    page_numbers: list[int] = []

    def resolve_doc_items(self, docling_document: "DoclingDocument") -> list["DocItem"]:
        """Resolve doc_item_refs to actual DocItem objects.

        Args:
            docling_document: The parent DoclingDocument containing the items.

        Returns:
            List of resolved DocItem objects. Items that fail to resolve are skipped.
        """
        from docling_core.types.doc.document import RefItem

        doc_items = []
        for ref in self.doc_item_refs:
            try:
                ref_item = RefItem.model_validate({"$ref": ref})
                doc_item = ref_item.resolve(docling_document)
                doc_items.append(doc_item)
            except Exception:
                # Graceful degradation: skip refs that can't be resolved
                continue
        return doc_items

    def resolve_bounding_boxes(
        self, docling_document: "DoclingDocument"
    ) -> list[BoundingBox]:
        """Resolve doc_item_refs to bounding boxes for visual grounding.

        Args:
            docling_document: The parent DoclingDocument containing the items.

        Returns:
            List of BoundingBox objects from resolved DocItems' provenance.
        """
        bounding_boxes = []
        for doc_item in self.resolve_doc_items(docling_document):
            prov = getattr(doc_item, "prov", None)
            if not prov:
                continue
            for prov_item in prov:
                bbox = getattr(prov_item, "bbox", None)
                if bbox is None:
                    continue
                bounding_boxes.append(
                    BoundingBox(
                        page_no=prov_item.page_no,
                        left=bbox.l,
                        top=bbox.t,
                        right=bbox.r,
                        bottom=bbox.b,
                    )
                )
        return bounding_boxes


class Chunk(BaseModel):
    """
    Represents a chunk with content, metadata, and optional document information.
    """

    id: str | None = None
    document_id: str | None = None
    content: str
    metadata: dict = {}
    order: int = 0
    document_uri: str | None = None
    document_title: str | None = None
    document_meta: dict = {}
    embedding: list[float] | None = None

    def get_chunk_metadata(self) -> ChunkMetadata:
        """Parse metadata dict into structured ChunkMetadata."""
        return ChunkMetadata.model_validate(self.metadata)


class SearchResult(BaseModel):
    """Search result with optional provenance information for citations."""

    content: str
    score: float
    chunk_id: str | None = None
    document_id: str | None = None
    document_uri: str | None = None
    document_title: str | None = None
    doc_item_refs: list[str] = []
    page_numbers: list[int] = []
    headings: list[str] | None = None
    labels: list[str] = []

    @classmethod
    def from_chunk(
        cls,
        chunk: "Chunk",
        score: float,
    ) -> "SearchResult":
        """Create from a Chunk."""
        meta = chunk.get_chunk_metadata()
        return cls(
            content=chunk.content,
            score=score,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_uri=chunk.document_uri,
            document_title=chunk.document_title,
            doc_item_refs=meta.doc_item_refs,
            page_numbers=meta.page_numbers,
            headings=meta.headings,
            labels=meta.labels,
        )

    def format_for_agent(self) -> str:
        """Format this search result for inclusion in agent context.

        Produces a structured format with metadata that helps LLMs understand
        the source and nature of the content.
        """
        parts = [f"[{self.chunk_id}] (score: {self.score:.2f})"]

        # Document source info
        source_parts = []
        if self.document_title:
            source_parts.append(f'"{self.document_title}"')
        if self.headings:
            source_parts.append(" > ".join(self.headings))
        if source_parts:
            parts.append(f"Source: {' > '.join(source_parts)}")

        # Content type (use primary label if available)
        if self.labels:
            primary_label = self._get_primary_label()
            if primary_label:
                parts.append(f"Type: {primary_label}")

        # The actual content
        parts.append(f"Content:\n{self.content}")

        return "\n".join(parts)

    def _get_primary_label(self) -> str | None:
        """Get the most significant label for display.

        Prioritizes structural labels over text labels.
        """
        if not self.labels:
            return None

        # Priority order: structural > contextual > text
        priority = {
            "table": 1,
            "code": 2,
            "form": 3,
            "key_value_region": 4,
            "list_item": 5,
            "formula": 6,
            "chart": 7,
            "picture": 8,
            "caption": 9,
            "footnote": 10,
            "section_header": 11,
            "title": 12,
        }

        # Find highest priority label
        best_label = None
        best_priority = float("inf")
        for label in self.labels:
            if label in priority and priority[label] < best_priority:
                best_label = label
                best_priority = priority[label]

        # Return best structural/special label, or first label if all are text
        return best_label if best_label else self.labels[0]
