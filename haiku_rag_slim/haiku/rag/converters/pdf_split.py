"""Split large PDFs into N-page slices, convert each, merge with
``DoclingDocument.concatenate``.

Rationale: docling's parser pipeline is memory-hungry and has confirmed leaks
(docling #2209, #1343, #2954; docling-serve #366, #474). Single-pass convert
of a 400-page PDF can OOM a workstation. Splitting bounds the peak working set
to one slice's pipeline state. Each slice round-trips through the existing
``DocumentConverter`` interface — so the docling-local and docling-serve
adapters work without modification, and each docling-serve task is independent
on the server side too.

Merge uses ``DoclingDocument.concatenate`` from docling-core ≥ 2.75. It
re-indexes ``self_ref`` values and shifts ``prov.page_no`` via an internal
``page_delta`` against ``_max_page`` (see
``docling_core/types/doc/document.py::_DocIndex.index``). No helper needed
on our side.

Cross-page references (named destinations, multi-page link annotations) are
dropped at the byte-level split — accepted loss; haiku.rag doesn't surface
them.
"""

import asyncio
import io
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.converters.base import DocumentConverter


def iter_pdf_slices(path: Path, slice_size: int) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(start_page, end_page_inclusive, pdf_bytes)`` for each
    ``slice_size``-page slice of ``path``. Page numbers are 1-based to match
    docling's ``prov.page_no`` convention. Each yielded byte string is a
    standalone PDF that docling can convert.
    """
    if slice_size <= 0:
        raise ValueError(f"slice_size must be >= 1, got {slice_size}")
    src = pdfium.PdfDocument(str(path))
    try:
        total = len(src)
        for start in range(0, total, slice_size):
            end = min(start + slice_size, total)
            dst = pdfium.PdfDocument.new()
            try:
                dst.import_pages(src, list(range(start, end)))
                buf = io.BytesIO()
                dst.save(buf)
                yield (start + 1, end, buf.getvalue())
            finally:
                dst.close()
    finally:
        src.close()


async def convert_pdf_with_splitting(
    converter: "DocumentConverter",
    path: Path,
    source_uri: str | None,
    slice_size: int,
) -> "DoclingDocument":
    """Split a PDF, convert each slice through ``converter``, return the
    merged ``DoclingDocument``.

    A single slice failure aborts the whole job — surfaced as ``ValueError``
    so the ingester pipeline's classifier maps it to ``TransientError`` and
    the queue retries the entire document. Per-slice retry would risk
    interleaved partial state with subsequent runs and is not worth the
    complexity.
    """
    from docling_core.types.doc.document import DoclingDocument

    slices = await asyncio.to_thread(lambda: list(iter_pdf_slices(path, slice_size)))

    converted: list[DoclingDocument] = []
    for start, end, pdf_bytes in slices:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            try:
                slice_doc = await converter.convert_file(
                    tmp_path, source_uri=source_uri
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to convert slice pages {start}-{end} of {path}: {exc}"
                ) from exc
            converted.append(slice_doc)
        finally:
            tmp_path.unlink(missing_ok=True)

    return DoclingDocument.concatenate(converted)
