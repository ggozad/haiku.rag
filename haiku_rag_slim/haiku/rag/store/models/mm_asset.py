from pydantic import BaseModel


class MMAsset(BaseModel):
    """A multimodal asset (typically a Docling PictureItem crop) stored in LanceDB."""

    id: str | None = None
    document_id: str

    # Docling pointer (e.g. "#/pictures/3") used to resolve provenance / ordering.
    doc_item_ref: str

    # Optional ordering anchor (index from DoclingDocument.iterate_items()).
    item_index: int | None = None

    # Provenance for layout-aware UX (visual grounding, grouping, etc.)
    page_no: int | None = None
    bbox: dict | None = None  # {"left":..,"top":..,"right":..,"bottom":..}

    # Text fields for display/debugging (NOT used as the embedding input for Phase 1).
    caption: str | None = None
    description: str | None = None
    metadata: dict = {}

    # Multimodal embedding vector (dimension is configured in Store via LanceDB schema)
    embedding: list[float] | None = None


class MMSearchResult(BaseModel):
    """Search result for multimodal assets."""

    asset_id: str
    document_id: str
    score: float

    doc_item_ref: str
    item_index: int | None = None
    page_no: int | None = None
    bbox: dict | None = None
    caption: str | None = None
    description: str | None = None
