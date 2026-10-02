"""Embedded-file extraction for PDFs.

Ingestion stores each file a PDF embeds (its ``/EmbeddedFiles`` table) as a
child document. These helpers are the single definition of a child's bytes,
name, URI, content type and MD5, so anything reproducing a child outside
ingestion (a metadata back-fill, say) gets exactly what ingestion stored.
"""

import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Maximum length of an attachment chain rooted at a top-level ingest. With
# value 3, a PDF whose attachments contain PDFs which themselves contain
# PDFs is fully ingested (3 levels); a fourth nested level logs a warning
# and is skipped.
MAX_ATTACHMENT_DEPTH = 3


@dataclass(frozen=True)
class PdfAttachment:
    """One embedded file, as ingestion stores it."""

    name: str
    data: bytes
    content_type: str
    content_hash: str


def attachment_uri(parent_uri: str, name: str) -> str:
    """The URI a child document is stored under. Nested attachments chain
    fragments, since ``parent_uri`` is itself an attachment URI."""
    return f"{parent_uri}#attachment={quote(name, safe='')}"


def extract_pdf_attachments(
    parent_body: bytes, parent_uri: str
) -> dict[str, PdfAttachment] | None:
    """Return the embedded files of the PDF ``parent_body``, keyed by child URI.
    Returns ``None`` when the PDF can't be opened.

    Every pdfium call is held under ``PDFIUM_LOCK`` (shared with page slicing)
    because libpdfium's global C state is not thread-safe; concurrent access
    from another worker corrupts it and then fails valid PDFs with "Data format
    error" until the process restarts. The call is blocking: run it off the
    event loop (``asyncio.to_thread``) from async code.
    """
    import pypdfium2 as pdfium

    from haiku.rag.converters.pdf_split import PDFIUM_LOCK

    with PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(parent_body)
        except pdfium.PdfiumError as exc:
            logger.warning(
                "Cannot scan %s for embedded attachments: %s", parent_uri, exc
            )
            return None
        try:
            attachments: dict[str, PdfAttachment] = {}
            for i in range(pdf.count_attachments()):
                att = pdf.get_attachment(i)
                name = att.get_name()
                # A malformed PDF can carry an attachment with an empty /F, so
                # this is real validation on untrusted input — it just needs a
                # hand-crafted file to reach, which no fixture here produces.
                if not name:  # pragma: no cover - needs a malformed PDF
                    continue
                data = bytes(att.get_data())
                attachments[attachment_uri(parent_uri, name)] = PdfAttachment(
                    name=name,
                    data=data,
                    content_type=(
                        mimetypes.guess_type(name)[0] or "application/octet-stream"
                    ),
                    content_hash=hashlib.md5(data, usedforsecurity=False).hexdigest(),
                )
            return attachments
        finally:
            pdf.close()
