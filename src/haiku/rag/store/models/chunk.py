from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a chunk with content, metadata, and optional document information.
    """

    id: int | None = None
    document_id: int | None = None
    content: str
    metadata: dict = {}
    document_uri: str | None = None
    document_meta: dict = {}
    embedding: list[float] | None = None
