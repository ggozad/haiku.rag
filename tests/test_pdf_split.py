"""Unit tests for haiku.rag.converters.pdf_split.

The integration test that pins split-and-merge against a real-PDF baseline
lives in tests/test_converters.py — that path requires docling installed and
is gated accordingly. These tests cover the byte-level split mechanism and
the docling-core concatenate contract in isolation.
"""

import io
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from haiku.rag.converters.pdf_split import (
    convert_pdf_with_splitting,
    iter_pdf_slices,
)


def _make_pdf(page_count: int, tmp_path: Path) -> Path:
    """Synthesize a minimal valid multi-page PDF with pypdfium2. Each page is
    an A4-sized blank. Returns the path."""
    doc = pdfium.PdfDocument.new()
    try:
        for _ in range(page_count):
            doc.new_page(width=595.0, height=842.0)  # A4 in points
        out = tmp_path / "synth.pdf"
        with open(out, "wb") as f:
            doc.save(f)
        return out
    finally:
        doc.close()


def test_iter_pdf_slices_partitions_pages(tmp_path):
    src = _make_pdf(7, tmp_path)
    slices = list(iter_pdf_slices(src, slice_size=3))

    assert [(s, e) for s, e, _ in slices] == [(1, 3), (4, 6), (7, 7)]

    page_counts = []
    for _, _, pdf_bytes in slices:
        d = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        try:
            page_counts.append(len(d))
        finally:
            d.close()
    assert page_counts == [3, 3, 1]
    assert sum(page_counts) == 7


def test_iter_pdf_slices_single_slice_when_doc_fits(tmp_path):
    src = _make_pdf(4, tmp_path)
    slices = list(iter_pdf_slices(src, slice_size=10))
    assert len(slices) == 1
    start, end, pdf_bytes = slices[0]
    assert (start, end) == (1, 4)
    d = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        assert len(d) == 4
    finally:
        d.close()


def test_iter_pdf_slices_rejects_zero_slice_size(tmp_path):
    src = _make_pdf(2, tmp_path)
    with pytest.raises(ValueError, match="slice_size must be >= 1"):
        list(iter_pdf_slices(src, slice_size=0))


@pytest.mark.asyncio
async def test_convert_unlinks_slice_tempfile_on_write_failure(tmp_path, monkeypatch):
    """If writing the slice bytes to the tempfile raises (e.g. ENOSPC), the
    tempfile is created on disk but never reaches the converter. The original
    error must surface and the orphaned file must be removed."""
    src = _make_pdf(2, tmp_path)

    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    import tempfile as _tempfile

    real_factory = _tempfile.NamedTemporaryFile
    created: list[Path] = []

    def _make_failing_tempfile(*args, **kwargs):
        handle = real_factory(*args, **kwargs)
        created.append(Path(handle.name))
        original_write = handle.write

        def _raising_write(_data):
            # Touch the underlying file once so we know the path exists on
            # disk and the cleanup actually has something to remove.
            original_write(b"\0")
            raise OSError("No space left on device")

        handle.write = _raising_write
        return handle

    monkeypatch.setattr(_tempfile, "NamedTemporaryFile", _make_failing_tempfile)

    class _UnusedConverter:
        async def convert_file(self, path: Path, *, source_uri):
            raise AssertionError("converter must not be reached on write failure")

    with pytest.raises(OSError, match="No space left on device"):
        await convert_pdf_with_splitting(
            _UnusedConverter(),  # ty: ignore[invalid-argument-type]
            src,
            source_uri=None,
            slice_size=1,
        )

    assert created, "expected NamedTemporaryFile to be called at least once"
    leftover = [p for p in created if p.exists()]
    assert leftover == [], f"tempfiles leaked after write failure: {leftover}"


@pytest.mark.asyncio
async def test_convert_aborts_and_cleans_up_on_mid_stream_slice_failure(
    tmp_path, monkeypatch
):
    """When converting slice 2 of 3 fails, the whole call must raise
    ValueError naming the failed slice's page range, every tempfile created
    along the way must be deleted, and the source PDF handle must be closed.
    """
    src = _make_pdf(7, tmp_path)

    # Pin tempfiles to a per-test dir so we can list leaks deterministically.
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    calls: list[Path] = []

    class _FlakyConverter:
        async def convert_file(self, path: Path, *, source_uri):
            calls.append(path)
            if len(calls) == 2:
                raise RuntimeError("docling exploded")
            from docling_core.types.doc.document import DoclingDocument

            return DoclingDocument(name="slice")

    with pytest.raises(ValueError, match="pages 4-6"):
        await convert_pdf_with_splitting(
            _FlakyConverter(),  # ty: ignore[invalid-argument-type]
            src,
            source_uri=None,
            slice_size=3,
        )

    # All converter inputs were under tmp_path (the pinned tempdir)…
    assert all(str(p).startswith(str(tmp_path)) for p in calls)
    # …and none of them are still on disk.
    leftover = [p for p in calls if p.exists()]
    assert leftover == [], f"tempfiles leaked: {leftover}"
    # We aborted after slice 2; slice 3 was never attempted.
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_concatenate_runs_off_event_loop_thread(tmp_path, monkeypatch):
    """DoclingDocument.concatenate merges slice documents that carry inlined
    base64 page/picture images — CPU-heavy and proportional to total document
    size. It must run off the event-loop thread so it doesn't stall other
    workers' coroutines. Capture the thread it runs on and assert it is not the
    main thread."""
    import threading

    from docling_core.types.doc.document import DoclingDocument

    src = _make_pdf(4, tmp_path)

    class _Converter:
        async def convert_file(self, path: Path, *, source_uri):
            return DoclingDocument(name="slice")

    called_from: list[threading.Thread] = []

    def spy(docs):
        called_from.append(threading.current_thread())
        # Return a slice doc rather than exercising the real concatenate —
        # this test only asserts the dispatch thread, not merge correctness
        # (covered by test_concatenate_shifts_page_nos_and_unique_self_refs).
        return docs[0]

    monkeypatch.setattr(DoclingDocument, "concatenate", staticmethod(spy))

    await convert_pdf_with_splitting(
        _Converter(),  # ty: ignore[invalid-argument-type]
        src,
        source_uri=None,
        slice_size=2,
    )

    assert called_from, "concatenate was never called"
    assert called_from[0] is not threading.main_thread(), (
        "DoclingDocument.concatenate ran on the event-loop thread; it must be "
        "dispatched via asyncio.to_thread"
    )


def test_concatenate_shifts_page_nos_and_unique_self_refs():
    """Pins the docling-core contract we rely on: when two docs (each with
    items on page 1) are concatenated, the second doc's items move to page 2
    and self_refs across both stay unique."""
    pytest.importorskip("docling_core")
    from docling_core.types.doc.base import BoundingBox, CoordOrigin
    from docling_core.types.doc.document import (
        DoclingDocument,
        PageItem,
        ProvenanceItem,
        Size,
    )
    from docling_core.types.doc.labels import DocItemLabel

    def _make_one_page_doc(name: str, text: str) -> DoclingDocument:
        d = DoclingDocument(name=name)
        d.pages[1] = PageItem(page_no=1, size=Size(width=595.0, height=842.0))
        d.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(
                    l=0.0, t=0.0, r=100.0, b=20.0, coord_origin=CoordOrigin.TOPLEFT
                ),
                charspan=(0, len(text)),
            ),
        )
        return d

    a = _make_one_page_doc("a", "alpha")
    b = _make_one_page_doc("b", "beta")

    merged = DoclingDocument.concatenate([a, b])

    assert len(merged.texts) == 2
    refs = [t.self_ref for t in merged.texts]
    assert len(set(refs)) == 2, f"self_refs collided: {refs}"

    page_nos = sorted({p.page_no for t in merged.texts for p in t.prov})
    assert page_nos == [1, 2], f"expected b's page 1 to shift to page 2, got {page_nos}"

    assert sorted(merged.pages.keys()) == [1, 2]
