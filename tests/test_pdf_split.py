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

from haiku.rag.converters.pdf_split import iter_pdf_slices


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
