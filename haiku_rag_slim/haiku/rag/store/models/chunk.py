from pydantic import BaseModel


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
