from pydantic import BaseModel


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
