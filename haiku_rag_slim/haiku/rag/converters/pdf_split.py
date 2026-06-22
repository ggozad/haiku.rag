import asyncio
import io
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.telemetry import logfire

# pypdfium2 wraps libpdfium, which has global C state and is not thread-safe.
# Two workers calling into pdfium concurrently race on that state and corrupt
# it — the first error surfaces as e.g. "Failed to import pages", and after
# that every subsequent PDF load fails with "Data format error" until the
# process restarts. This is the single process-wide lock around *all* in-process
# pdfium access (page slicing here and embedded-attachment scanning in
# client.documents); every pdfium call must hold it so only one runs at a time.
# Slicing releases it between slices so other callers can interleave; the heavy
# work (docling convert) happens between yields with the lock free.
PDFIUM_LOCK = threading.Lock()

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.converters.base import DocumentConverter

_SENTINEL: object = object()


def iter_pdf_slices(
    path: Path, slice_size: int
) -> Generator[tuple[int, int, bytes], None, None]:
    """Yield ``(start_page, end_page_inclusive, pdf_bytes)`` for each
    ``slice_size``-page slice of ``path``. Page numbers are 1-based to match
    docling's ``prov.page_no`` convention. Each yielded byte string is a
    standalone PDF that docling can convert.

    The pdfium lock is held only around each pdfium call (open, slice
    extraction, close) — never across a ``yield``. The source ``PdfDocument``
    handle stays open across yields; per-document handles coexist safely as
    long as no two pdfium calls execute concurrently.
    """
    if slice_size <= 0:
        raise ValueError(f"slice_size must be >= 1, got {slice_size}")
    with PDFIUM_LOCK:
        try:
            src = pdfium.PdfDocument(str(path))
        except pdfium.PdfiumError as exc:
            raise UnsupportedSourceError(
                f"pypdfium2 cannot open PDF {path}: {exc}"
            ) from exc
        total = len(src)
    try:
        for start in range(0, total, slice_size):
            end = min(start + slice_size, total)
            with PDFIUM_LOCK:
                dst = pdfium.PdfDocument.new()
                try:
                    dst.import_pages(src, list(range(start, end)))
                    buf = io.BytesIO()
                    dst.save(buf)
                    slice_bytes = buf.getvalue()
                finally:
                    dst.close()
            yield (start + 1, end, slice_bytes)
    finally:
        with PDFIUM_LOCK:
            src.close()


async def convert_pdf_with_splitting(
    converter: "DocumentConverter",
    path: Path,
    source_uri: str | None,
    slice_size: int,
) -> "DoclingDocument":
    """Split a PDF, convert each slice through ``converter``, return the
    merged ``DoclingDocument``.

    Slices are produced lazily — only one slice's bytes live in memory at a
    time, so peak working set stays bounded regardless of page count. A
    single slice failure aborts the whole job (raised as ``ValueError`` so
    the ingester pipeline classifies it as ``TransientError`` and the queue
    retries the entire document). Per-slice retry would risk interleaved
    partial state with subsequent runs and is not worth the complexity.
    """
    from docling_core.types.doc.document import DoclingDocument

    def _next(it):
        return next(it, _SENTINEL)

    it = iter_pdf_slices(path, slice_size)
    converted: list[DoclingDocument] = []
    try:
        while True:
            slice_item = await asyncio.to_thread(_next, it)
            if slice_item is _SENTINEL:
                break
            start, end, pdf_bytes = slice_item
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".pdf", delete=False
                ) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(pdf_bytes)
                    tmp.flush()
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
                # `delete=False` is required so the converter (which opens
                # tmp_path itself) sees a fully written, closed file. Unlink
                # in finally so a mid-write disk-full / mid-convert failure
                # doesn't leak the slice on disk.
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
    finally:
        # Close the generator under the pdfium lock so src.close() runs
        # even when we abort mid-stream (slice failure, cancellation).
        # Off the event loop because the close path acquires the lock.
        await asyncio.to_thread(it.close)

    # Merge off the event loop: concatenating slice documents that carry
    # inlined base64 page/picture images is CPU-heavy and proportional to the
    # total document size, so running it inline would block other coroutines.
    return await asyncio.to_thread(DoclingDocument.concatenate, converted)
