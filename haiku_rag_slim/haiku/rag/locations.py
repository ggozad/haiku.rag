from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.store.models.chunk import Chunk

LOCATIONS_KEY = "locations"
"""Chunk metadata key of the chunk's boxes, per page."""


def item_locations(
    document: "DoclingDocument", refs: list[str]
) -> list[dict[str, Any]]:
    """Where the items `refs` sit, page by page in page order.

    Each page is `{"page", "width", "height", "boxes"}`, every box
    `[left, top, right, bottom]` in the page's units with the origin at its
    top-left corner. Items without provenance, and refs that do not resolve,
    are left out.
    """
    from docling_core.types.doc.document import RefItem

    pages: dict[int, dict[str, Any]] = {}
    for ref in refs:
        try:
            item = RefItem.model_validate({"$ref": ref}).resolve(document)
        except (AttributeError, IndexError, ValueError):
            continue
        for prov in getattr(item, "prov", None) or []:
            page = document.pages.get(prov.page_no)
            if page is None:
                continue
            box = prov.bbox.to_top_left_origin(page.size.height)
            entry = pages.setdefault(
                prov.page_no,
                {
                    "page": prov.page_no,
                    "width": round(page.size.width, 2),
                    "height": round(page.size.height, 2),
                    "boxes": [],
                },
            )
            entry["boxes"].append(
                [round(box.l, 2), round(box.t, 2), round(box.r, 2), round(box.b, 2)]
            )
    return [pages[number] for number in sorted(pages)]


def attach_locations(document: "DoclingDocument", chunks: "list[Chunk]") -> None:
    """Record each chunk's boxes in its metadata under `LOCATIONS_KEY`."""
    for chunk in chunks:
        refs = (chunk.metadata or {}).get("doc_item_refs") or []
        if locations := item_locations(document, refs):
            chunk.metadata = {**chunk.metadata, LOCATIONS_KEY: locations}
