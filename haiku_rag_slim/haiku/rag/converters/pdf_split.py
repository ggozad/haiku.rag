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
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import logfire
import pypdfium2 as pdfium

from haiku.rag.client.exceptions import UnsupportedSourceError

# pypdfium2 wraps libpdfium, which has global C state and is not thread-safe.
# Multiple workers calling iter_pdf_slices concurrently (via asyncio.to_thread)
# race on that state and corrupt it — first error surfaces as e.g.
# "Failed to import pages", and after that every subsequent PDF load fails
# with "Failed to load document" until the process restarts. Hold this lock
# around every pdfium call so only one worker's slicing runs at a time. The
# actual heavy work (docling-serve convert) happens AFTER iter_pdf_slices
# returns and continues to parallelize across workers.
_PDFIUM_LOCK = threading.Lock()

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
    with _PDFIUM_LOCK:
        try:
            src = pdfium.PdfDocument(str(path))
        except pdfium.PdfiumError as exc:
            # Malformed / unreadable PDF — retrying won't help. Raise the
            # typed error so the ingester pipeline classifies it as
            # PermanentError and dead-letters the job instead of cycling it
            # through attempts.
            raise UnsupportedSourceError(
                f"pypdfium2 cannot open PDF {path}: {exc}"
            ) from exc
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
            # One span per slice so each docling-serve round-trip (or local
            # docling invocation) is visible in Logfire; the outer
            # `document.convert` span aggregates the lot.
            with logfire.span(
                "document.convert_slice",
                uri=source_uri,
                start_page=start,
                end_page=end,
            ):
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
