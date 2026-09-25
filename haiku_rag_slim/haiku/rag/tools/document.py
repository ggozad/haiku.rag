from pydantic import BaseModel


class DocumentInfo(BaseModel):
    """Document info for list_documents response."""

    id: str | None = None
    title: str
    uri: str
    created: str
    source: str | None = None
    metadata: dict = {}


class OutlineNode(BaseModel):
    """A heading in a document's outline. `id` is the heading item's self_ref."""

    id: str
    title: str
    level: int
    page_numbers: list[int] = []
    children: list["OutlineNode"] = []


class DocumentSection(BaseModel):
    """One section's text in reading order, subsections included."""

    id: str
    title: str
    page_numbers: list[int] = []
    content: str
