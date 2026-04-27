from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, NodeItem


class DocumentItem(BaseModel):
    document_id: str
    position: int
    self_ref: str
    label: str = ""
    text: str = ""
    page_numbers: list[int] = []
    picture_data: bytes | None = None


def extract_item_text(item: "NodeItem", docling_doc: "DoclingDocument") -> str | None:
    """Extract text content from a DocItem.

    Handles different item types:
    - TextItem, SectionHeaderItem, etc.: Use .text attribute
    - TableItem: Use export_to_markdown() for table content
    - PictureItem: Use export_to_markdown() with PLACEHOLDER mode to avoid base64
    """
    from docling_core.types.doc.base import ImageRefMode
    from docling_core.types.doc.document import PictureItem, TableItem

    if text := getattr(item, "text", None):
        return text

    if isinstance(item, PictureItem):
        return item.export_to_markdown(
            docling_doc,
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="",
        )

    if isinstance(item, TableItem):
        try:
            return item.export_to_markdown(docling_doc)
        except Exception:
            pass

    if caption := getattr(item, "caption", None):
        if hasattr(caption, "text"):
            return caption.text

    return None


def extract_items(
    document_id: str, docling_doc: "DoclingDocument"
) -> list[DocumentItem]:
    """Extract document items from a DoclingDocument for the items table.

    Runs iterate_items() and extracts the fields needed for context expansion:
    self_ref, label, pre-rendered text, and page numbers from provenance.
    Items are stored as docling produces them — container items (e.g., list_item)
    may have empty text with content in their children.
    """
    items: list[DocumentItem] = []

    for position, (item, _level) in enumerate(docling_doc.iterate_items()):
        label = getattr(item, "label", None)
        label_str = str(label.value) if hasattr(label, "value") else str(label or "")

        text = extract_item_text(item, docling_doc) or ""

        page_numbers: list[int] = []
        if prov := getattr(item, "prov", None):
            for p in prov:
                page_no = getattr(p, "page_no", None)
                if page_no is not None and page_no not in page_numbers:
                    page_numbers.append(page_no)

        items.append(
            DocumentItem(
                document_id=document_id,
                position=position,
                self_ref=item.self_ref,
                label=label_str,
                text=text,
                page_numbers=sorted(page_numbers),
            )
        )

    return items
