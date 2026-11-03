"""A conversion process speaking the supervisor's protocol, with documents
that misbehave on cue: by its name, a file hangs, crashes or fails."""

import os
import time

from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.converters.supervisor_worker import main


def convert(config, path, source_uri, page_range, ocr_engine):
    name = path.name
    engine = ocr_engine or config.processing.conversion_options.ocr_engine
    if name.startswith("hangpage"):
        if page_range and page_range[0] <= 4 <= page_range[1]:
            time.sleep(3600)
    elif name.startswith("hang-always") or (
        name.startswith("hang") and engine != "rapidocr"
    ):
        time.sleep(3600)
    if name.startswith("slow"):
        time.sleep(1.0)
    if name.startswith("crash"):
        os._exit(3)
    if name.startswith("bad"):
        raise ValueError("cannot parse")
    doc = DoclingDocument(name=path.stem)
    if page_range is None:
        doc.add_text(label=DocItemLabel.TEXT, text=f"{name} by {engine}")
        return doc
    first, last = page_range
    for page_no in range(first, last + 1):
        # The last page of the first batch goes missing, as a page docling
        # cannot load does.
        if first == 1 and page_no == last and last > 1:
            continue
        doc.add_page(page_no=page_no, size=Size(width=100, height=200))
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"page {page_no}",
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=1, t=2, r=3, b=4),
                charspan=(0, 6),
            ),
        )
    return doc


if __name__ == "__main__":
    main(convert)
