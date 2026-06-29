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


def _incremental_merge_available() -> bool:
    """Whether docling-core exposes the internal merge buffer this module
    drives incrementally. False on versions that renamed/removed it, in which
    case ``_SliceMerger`` falls back to the public ``concatenate``."""
    from docling_core.types.doc.document import DoclingDocument

    return hasattr(DoclingDocument, "_DocIndex") and hasattr(
        DoclingDocument, "_update_from_index"
    )


class _SliceMerger:
    """Merges converted PDF-slice documents into one ``DoclingDocument``.

    ``DoclingDocument.concatenate`` indexes each input into an internal
    ``_DocIndex`` merge buffer (deep-copying its items), then materialises the
    result. Collecting every slice up front and concatenating at the end thus
    holds all slices *and* the merged copy at once — roughly twice the
    document. This merger drives the same ``_DocIndex`` incrementally: each
    slice is folded in and released before the next is converted, so peak
    working set is ~one merged document plus one slice. The sequence of
    ``index()`` calls is identical to ``concatenate``'s, so the merged result
    is byte-for-byte the same.

    ``_DocIndex``/``_update_from_index`` are docling-core internals; when they
    are unavailable (version drift, per ``_incremental_merge_available``) the
    merger collects slices and calls the public ``concatenate`` instead.
    """

    def __init__(self) -> None:
        from docling_core.types.doc.document import DoclingDocument

        self._incremental = _incremental_merge_available()
        self._index = DoclingDocument._DocIndex() if self._incremental else None
        self._collected: list[DoclingDocument] = []

    def add(self, slice_doc: "DoclingDocument") -> None:
        """Fold one slice into the merge. CPU-bound (deep-copies the slice's
        items), so call via ``asyncio.to_thread`` to keep it off the loop."""
        if self._incremental:
            assert self._index is not None
            self._index.index(slice_doc)
        else:
            self._collected.append(slice_doc)

    def result(self) -> "DoclingDocument":
        """Materialise the merged document. Call via ``asyncio.to_thread``."""
        from docling_core.types.doc.document import DoclingDocument

        if self._incremental:
            assert self._index is not None
            merged = DoclingDocument(name="")
            merged._update_from_index(self._index)
            return merged
        return DoclingDocument.concatenate(self._collected)


async def convert_pdf_with_splitting(
    converter: "DocumentConverter",
    path: Path,
    source_uri: str | None,
    slice_size: int,
) -> "DoclingDocument":
    """Split a PDF, convert each slice through ``converter``, return the
    merged ``DoclingDocument``.

    Slices are produced lazily — only one slice's bytes live in memory at a
    time — and each converted slice is folded into the running merge and
    released before the next is converted (see ``_SliceMerger``), so peak
    working set is ~one merged document plus one slice rather than every slice
    at once. A single slice failure aborts the whole job (raised as
    ``ValueError`` so the ingester pipeline classifies it as ``TransientError``
    and the queue retries the entire document). Per-slice retry would risk
    interleaved partial state with subsequent runs and is not worth the
    complexity.
    """

    def _next(it):
        return next(it, _SENTINEL)

    it = iter_pdf_slices(path, slice_size)
    merger = _SliceMerger()
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
                # Fold the slice in and drop it before the next slice converts,
                # so peak memory holds ~one merged doc + one slice rather than
                # every slice at once. index() deep-copies, so run it off the
                # event loop.
                await asyncio.to_thread(merger.add, slice_doc)
                del slice_doc
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

    # Materialise off the event loop: the merge buffer holds inlined base64
    # page/picture images, so building the result is CPU-heavy and proportional
    # to total document size — running it inline would block other coroutines.
    return await asyncio.to_thread(merger.result)
