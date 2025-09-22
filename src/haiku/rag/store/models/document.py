from datetime import datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a document with an ID, content, and metadata.
    """

    id: str | None = None
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
