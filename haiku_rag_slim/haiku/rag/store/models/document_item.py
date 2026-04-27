import base64
import binascii
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, NodeItem, PictureItem


class DocumentItem(BaseModel):
    document_id: str
    position: int
    self_ref: str
    label: str = ""
    text: str = ""
    page_numbers: list[int] = []
    picture_data: bytes | None = None


def _picture_description_text(item: "PictureItem") -> str | None:
    """Return the VLM-generated description text for a PictureItem, if any.

    Tries the modern ``meta.description`` location first (docling 2.91+) and
    falls back to ``annotations`` entries that carry a ``text`` field
    (PictureDescriptionData and similar).
    """
    meta = getattr(item, "meta", None)
    if meta is not None:
        description = getattr(meta, "description", None)
        if description is not None:
            text = getattr(description, "text", None)
            if isinstance(text, str) and text.strip():
                return text
    annotations = getattr(item, "annotations", None) or []
    for ann in annotations:
        text = getattr(ann, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    return None


def _decode_picture_bytes(item: "PictureItem") -> bytes | None:
    """Decode a PictureItem's embedded image into raw bytes.

    Reads ``item.image.uri`` and base64-decodes it when it is a ``data:`` URI.
    Returns None for items whose image is absent, stripped, or not a data URI
    (e.g. file references). Tolerant of malformed data — returns None on any
    decode failure rather than raising.
    """
    image = getattr(item, "image", None)
    if image is None:
        return None
    uri = getattr(image, "uri", None)
    if uri is None:
        return None
    uri_str = str(uri)
    if not uri_str.startswith("data:"):
        return None
    try:
        _, encoded = uri_str.split(",", 1)
    except ValueError:
        return None
    try:
        return base64.b64decode(encoded, validate=False)
    except (ValueError, binascii.Error):
        return None


def extract_item_text(item: "NodeItem", docling_doc: "DoclingDocument") -> str | None:
    """Extract text content from a DocItem.

    Handles different item types:
    - TextItem, SectionHeaderItem, etc.: Use .text attribute
    - TableItem: Use export_to_markdown() for table content
    - PictureItem: Prefer the VLM description (when picture_description is on)
      so pictures carry meaningful prose into chunk text and survive
      ``expand_with_items``' ``if item.text:`` filter; otherwise fall back to
      a placeholder markdown export (no base64).
    """
    from docling_core.types.doc.base import ImageRefMode
    from docling_core.types.doc.document import PictureItem, TableItem

    if text := getattr(item, "text", None):
        return text

    if isinstance(item, PictureItem):
        if description := _picture_description_text(item):
            return description
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
    document_id: str,
    docling_doc: "DoclingDocument",
    existing_picture_data: dict[str, bytes] | None = None,
) -> list[DocumentItem]:
    """Extract document items from a DoclingDocument for the items table.

    Runs iterate_items() and extracts the fields needed for context expansion:
    self_ref, label, pre-rendered text, and page numbers from provenance.
    Items are stored as docling produces them — container items (e.g., list_item)
    may have empty text with content in their children.

    For PictureItems, the embedded image is decoded from ``image.uri`` (a base64
    data URI) and stored on ``DocumentItem.picture_data``. When the live docling
    has already had its picture URIs stripped (rebuild / re-extract scenarios
    where the docling structure round-trips through the compressed blob), a
    fall-back lookup against ``existing_picture_data`` (keyed by ``self_ref``)
    preserves the bytes that were captured at original ingest time.
    """
    from docling_core.types.doc.document import PictureItem

    existing = existing_picture_data or {}
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

        picture_data: bytes | None = None
        if isinstance(item, PictureItem):
            picture_data = _decode_picture_bytes(item)
            if picture_data is None:
                picture_data = existing.get(item.self_ref)

        items.append(
            DocumentItem(
                document_id=document_id,
                position=position,
                self_ref=item.self_ref,
                label=label_str,
                text=text,
                page_numbers=sorted(page_numbers),
                picture_data=picture_data,
            )
        )

    return items
