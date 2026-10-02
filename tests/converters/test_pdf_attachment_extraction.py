import hashlib
import io
import logging

import pypdfium2 as pdfium

from haiku.rag.client.documents import _extract_pdf_attachments
from haiku.rag.converters.pdf_attachments import (
    MAX_ATTACHMENT_DEPTH,
    PdfAttachment,
    attachment_uri,
    extract_pdf_attachments,
)
from haiku.rag.converters.pdf_split import PDFIUM_LOCK
from tests.conftest import capture_logs


def build_pdf(attachments: list[tuple[str, bytes]]) -> bytes:
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 200)
    for name, data in attachments:
        pdf.new_attachment(name).set_data(data)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def test_attachment_uri_percent_encodes_the_name():
    assert (
        attachment_uri("file:///p.pdf", "memo with spaces.pdf")
        == "file:///p.pdf#attachment=memo%20with%20spaces.pdf"
    )
    assert attachment_uri("file:///p.pdf", "100%/a#b.txt") == (
        "file:///p.pdf#attachment=100%25%2Fa%23b.txt"
    )


def test_attachment_uri_chains_for_nested_attachments():
    child = attachment_uri("file:///p.pdf", "l1.pdf")
    assert attachment_uri(child, "l2.pdf") == (
        "file:///p.pdf#attachment=l1.pdf#attachment=l2.pdf"
    )


def test_extract_returns_attachments_keyed_by_child_uri():
    body = build_pdf([("a b.txt", b"A"), ("50%.pdf", b"P"), ("blob", b"X")])

    attachments = extract_pdf_attachments(body, "file:///p.pdf")

    assert attachments == {
        "file:///p.pdf#attachment=a%20b.txt": PdfAttachment(
            name="a b.txt",
            data=b"A",
            content_type="text/plain",
            content_hash=hashlib.md5(b"A", usedforsecurity=False).hexdigest(),
        ),
        "file:///p.pdf#attachment=50%25.pdf": PdfAttachment(
            name="50%.pdf",
            data=b"P",
            content_type="application/pdf",
            content_hash=hashlib.md5(b"P", usedforsecurity=False).hexdigest(),
        ),
        "file:///p.pdf#attachment=blob": PdfAttachment(
            name="blob",
            data=b"X",
            content_type="application/octet-stream",
            content_hash=hashlib.md5(b"X", usedforsecurity=False).hexdigest(),
        ),
    }


def test_extract_of_a_pdf_without_attachments_is_empty():
    assert extract_pdf_attachments(build_pdf([]), "file:///p.pdf") == {}


def test_extract_of_an_unopenable_pdf_returns_none():
    logger = logging.getLogger("haiku.rag.converters.pdf_attachments")
    with capture_logs(logger, logging.WARNING) as records:
        assert extract_pdf_attachments(b"not a pdf", "file:///junk.pdf") is None
    assert "file:///junk.pdf" in records[0].getMessage()


def test_extract_holds_the_pdfium_lock_and_closes_the_document(monkeypatch):
    """libpdfium is not thread-safe: every call runs under the process-wide
    lock, and the document handle is released before the lock is."""
    body = build_pdf([("a.txt", b"A")])
    events: list[tuple[str, bool]] = []
    real = pdfium.PdfDocument

    class Spy(real):
        def __init__(self, *args, **kwargs):
            events.append(("open", PDFIUM_LOCK.locked()))
            super().__init__(*args, **kwargs)

        def count_attachments(self):
            events.append(("count", PDFIUM_LOCK.locked()))
            return super().count_attachments()

        def close(self):
            events.append(("close", PDFIUM_LOCK.locked()))
            super().close()

    monkeypatch.setattr(pdfium, "PdfDocument", Spy)

    extract_pdf_attachments(body, "file:///p.pdf")

    assert events == [("open", True), ("count", True), ("close", True)]
    assert not PDFIUM_LOCK.locked()


def test_private_wrapper_stops_at_the_depth_cap():
    body = build_pdf([("a.txt", b"A")])
    logger = logging.getLogger("haiku.rag.client.documents")

    assert _extract_pdf_attachments(body, "file:///p.pdf", depth=0) == (
        extract_pdf_attachments(body, "file:///p.pdf")
    )
    with capture_logs(logger, logging.WARNING) as records:
        assert (
            _extract_pdf_attachments(
                body, "file:///p.pdf", depth=MAX_ATTACHMENT_DEPTH - 1
            )
            is None
        )
    assert len(records) == 1


def test_private_wrapper_at_the_cap_is_quiet_without_attachments():
    logger = logging.getLogger("haiku.rag.client.documents")
    with capture_logs(logger, logging.WARNING) as records:
        assert (
            _extract_pdf_attachments(
                build_pdf([]), "file:///p.pdf", depth=MAX_ATTACHMENT_DEPTH - 1
            )
            is None
        )
    assert records == []


def test_private_wrapper_returns_none_for_an_unopenable_pdf():
    assert _extract_pdf_attachments(b"junk", "file:///p.pdf", depth=0) is None
