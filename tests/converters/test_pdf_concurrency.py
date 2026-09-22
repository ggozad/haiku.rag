import io
import threading

import pypdfium2 as pdfium

from haiku.rag.client.documents import _extract_pdf_attachments
from haiku.rag.converters.pdf_split import iter_pdf_slices


def _make_pdf(pages: int, attachment: tuple[str, bytes] | None) -> bytes:
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(200, 200)
    if attachment is not None:
        name, data = attachment
        pdf.new_attachment(name).set_data(data)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def test_concurrent_pdfium_access_does_not_corrupt_global_state(tmp_path):
    """libpdfium has global, non-thread-safe C state. The attachment scan and
    the page slicer both call into it; without a single shared lock across both
    sites, concurrent workers corrupt that state and then fail otherwise-valid
    PDFs with "Data format error". Both paths operate on valid PDFs here, so any
    failure means the global state was corrupted by a concurrent caller."""
    body = _make_pdf(pages=6, attachment=("notes.txt", b"payload"))
    path = tmp_path / "doc.pdf"
    path.write_bytes(body)

    scan_results: list[dict | None] = []
    slice_errors: list[str] = []
    lock = threading.Lock()

    def scan() -> None:
        for _ in range(40):
            r = _extract_pdf_attachments(body, "file:///doc.pdf", depth=0)
            with lock:
                scan_results.append(r)

    def slice_() -> None:
        for _ in range(40):
            try:
                slices = list(iter_pdf_slices(path, 2))
                assert len(slices) == 3
            except Exception as exc:  # noqa: BLE001
                with lock:
                    slice_errors.append(repr(exc))

    threads = [threading.Thread(target=scan) for _ in range(4)]
    threads += [threading.Thread(target=slice_) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed_scans = sum(r is None or len(r) != 1 for r in scan_results)
    assert failed_scans == 0, (
        f"{failed_scans}/{len(scan_results)} attachment scans failed"
    )
    assert slice_errors == [], slice_errors
