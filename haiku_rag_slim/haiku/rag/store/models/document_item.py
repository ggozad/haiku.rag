import base64
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, NodeItem, PictureItem


PICTURE_REF_PREFIX = "#/pictures/"


class DocumentItem(BaseModel):
    document_id: str
    position: int
    self_ref: str
    label: str = ""
    text: str = ""
    page_numbers: list[int] = []
    picture_data: bytes | None = None
    heading_level: int = 0
    tree_depth: int = 0


def _picture_description_text(item: "PictureItem") -> str | None:
    """Return the VLM-generated description text for a PictureItem, if any.

    Reads ``meta.description.text``. Pre-2.91 blobs that stored the
    description under the deprecated ``annotations`` field are migrated to
    ``meta`` by ``PictureItem``'s own ``@model_validator(mode="after")`` at
    load time, so this single check covers both formats.
    """
    if item.meta and item.meta.description:
        text = item.meta.description.text
        if text and text.strip():
            return text
    return None


def _decode_picture_bytes(item: "PictureItem") -> bytes | None:
    """Decode a PictureItem's embedded image into raw bytes.

    Reads ``item.image.uri`` and base64-decodes it when it is a ``data:`` URI.
    Returns None for items whose image is absent or stripped, or whose URI is
    a file reference rather than inline data.
    """
    if item.image is None:
        return None
    uri = str(item.image.uri)
    if not uri.startswith("data:"):
        return None
    _, encoded = uri.split(",", 1)
    return base64.b64decode(encoded, validate=False)


def _picture_caption_text(item: "PictureItem", docling_doc: "DoclingDocument") -> str:
    """Join a picture's caption texts with spaces, preserving word boundaries."""
    return " ".join(
        text
        for caption in item.captions
        if (text := caption.resolve(docling_doc).text.strip())
    )


def extract_item_text(
    item: "NodeItem",
    docling_doc: "DoclingDocument",
    *,
    get_serializer: Callable[[], Any] | None = None,
) -> str | None:
    """Extract text content from a DocItem.

    Handles different item types:
    - TextItem, SectionHeaderItem, etc.: Use .text attribute
    - TableItem: serialize to markdown. ``get_serializer`` supplies a reused
      ``MarkdownDocSerializer`` (see ``extract_items``); when absent a one-off
      serializer is built so direct calls keep working.
    - PictureItem: Prefer the VLM description (when picture_description is on)
      so pictures carry meaningful prose into chunk text and survive
      ``expand_with_items``' ``if item.text:`` filter; otherwise fall back to
      the picture's caption text.
    """
    from docling_core.types.doc.document import PictureItem, TableItem

    if text := getattr(item, "text", None):
        return text

    if isinstance(item, PictureItem):
        if description := _picture_description_text(item):
            return description
        return _picture_caption_text(item, docling_doc)

    if isinstance(item, TableItem):
        try:
            if get_serializer is None:
                from docling_core.transforms.serializer.markdown import (
                    MarkdownDocSerializer,
                )

                serializer = MarkdownDocSerializer(doc=docling_doc)
            else:
                serializer = get_serializer()
            return serializer.serialize(item=item).text
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
    from docling_core.types.doc.document import PictureItem, SectionHeaderItem

    existing = existing_picture_data or {}
    items: list[DocumentItem] = []

    serializer: Any = None

    def get_serializer() -> Any:
        nonlocal serializer
        if serializer is None:
            from docling_core.transforms.serializer.markdown import (
                MarkdownDocSerializer,
            )

            serializer = MarkdownDocSerializer(doc=docling_doc)
        return serializer

    for position, (item, level) in enumerate(docling_doc.iterate_items()):
        label = getattr(item, "label", None)
        label_str = str(label.value) if hasattr(label, "value") else str(label or "")

        text = extract_item_text(item, docling_doc, get_serializer=get_serializer) or ""

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

        heading_level = item.level if isinstance(item, SectionHeaderItem) else 0

        items.append(
            DocumentItem(
                document_id=document_id,
                position=position,
                self_ref=item.self_ref,
                label=label_str,
                text=text,
                page_numbers=sorted(page_numbers),
                picture_data=picture_data,
                heading_level=heading_level,
                tree_depth=level,
            )
        )

    return items
