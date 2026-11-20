from io import BytesIO
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from haiku.rag.config import AppConfig
from haiku.rag.converters.pdf_split import convert_pdf_with_splitting
from haiku.rag.utils.images import render_original_pages
from tests.platform import converts_pdf_locally

pytestmark = [
    pytest.mark.slow,
    converts_pdf_locally,
    pytest.mark.usefixtures("docling_local_models"),
]

DOCLAYNET = Path(__file__).parent.parent / "data" / "doclaynet.pdf"
SHIFT = 6


def _pdf_object(objects: list[bytes]) -> bytes:
    out = BytesIO()
    out.write(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, xref)
    )
    return out.getvalue()


def _stream(data: bytes, extra: bytes = b"") -> bytes:
    return b"<< /Length %d %s >>\nstream\n" % (len(data), extra) + data + b"\nendstream"


def _annotated_pdf(widget: bool) -> bytes:
    """A page of text with a form field's widget, or a square annotation,
    each drawn by its appearance stream."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    if widget:
        catalog = b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [5 0 R] >> >>"
        annotation = (
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (name) /V (VALUE) "
            b"/Rect [50 100 250 140] /F 4 /P 3 0 R /AP << /N 7 0 R >> >>"
        )
        appearance = b"0.2 0.2 0.8 rg 0 0 200 40 re f"
    else:
        catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
        annotation = (
            b"<< /Type /Annot /Subtype /Square /Rect [50 100 250 140] /F 4 "
            b"/C [1 0 0] /AP << /N 7 0 R >> >>"
        )
        appearance = b"1 0 0 RG 6 w 3 3 194 34 re S"
    return _pdf_object(
        [
            catalog,
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R "
            b"/Annots [5 0 R] >>",
            font,
            annotation,
            _stream(b"BT /F1 24 Tf 30 250 Td (Body text) Tj ET"),
            _stream(appearance, b"/Type /XObject /Subtype /Form /BBox [0 0 200 40]"),
        ]
    )


def _geometry_variants() -> bytes:
    """Pages of the DocLayNet paper under every page geometry docling and
    pdfium could disagree on, and a page carrying only a scanned image."""
    source = pdfium.PdfDocument(str(DOCLAYNET))
    pdf = pdfium.PdfDocument.new()
    pdf.import_pages(source, [0, 0, 0, 0, 0, 0, 0])
    plain, r90, r180, r270, cropped, shifted, cropped_rotated = (
        pdf[i] for i in range(7)
    )
    del plain
    r90.set_rotation(90)
    r180.set_rotation(180)
    r270.set_rotation(270)
    cropped.set_cropbox(60, 80, 520, 700)
    width, height = shifted.get_size()
    shifted.set_mediabox(-40, -30, width - 40, height - 30)
    cropped_rotated.set_cropbox(60, 80, 520, 700)
    cropped_rotated.set_rotation(90)

    scan = source[0].render(scale=1.5).to_pil()
    page = pdf.new_page(width, height)
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(scan))
    image.set_matrix(pdfium.PdfMatrix().scale(width, height))
    page.insert_obj(image)
    page.gen_content()

    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def _best_shift(expected: Image.Image, actual: Image.Image) -> tuple[int, int]:
    """The offset of `actual` against `expected` with the least difference."""
    a = np.asarray(expected.convert("L"), dtype=np.float32)
    b = np.asarray(actual.convert("L"), dtype=np.float32)
    h, w = a.shape
    inner = (slice(SHIFT, h - SHIFT), slice(SHIFT, w - SHIFT))
    scores = {
        (dx, dy): float(
            np.abs(
                a[inner] - b[SHIFT + dy : h - SHIFT + dy, SHIFT + dx : w - SHIFT + dx]
            ).mean()
        )
        for dx in range(-SHIFT, SHIFT + 1)
        for dy in range(-SHIFT, SHIFT + 1)
    }
    return min(scores, key=scores.__getitem__)


@pytest.fixture(autouse=True)
def docling_rasters(monkeypatch):
    """docling-local with page images on: the raster renders are compared to."""
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    build = DoclingLocalConverter._build_pipeline_options

    def with_page_images(self):
        options = build(self)
        options.generate_page_images = True
        return options

    monkeypatch.setattr(
        DoclingLocalConverter, "_build_pipeline_options", with_page_images
    )


def _config() -> AppConfig:
    return AppConfig()


def _raster(page) -> Image.Image:
    assert page.image is not None and page.image.pil_image is not None
    return page.image.pil_image


def _assert_renders_like(docling_doc, original: bytes, scale: float) -> None:
    rendered = render_original_pages(original, sorted(docling_doc.pages), scale)
    assert sorted(rendered) == sorted(docling_doc.pages)
    for page_no, page in docling_doc.pages.items():
        raster = _raster(page)
        assert rendered[page_no].size == raster.size, page_no
        assert _best_shift(raster, rendered[page_no]) == (0, 0), page_no


@pytest.mark.parametrize("slice_size", [None, 2])
async def test_pdf_pages_render_in_docling_frame(tmp_path, slice_size):
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    original = _geometry_variants()
    path = tmp_path / "variants.pdf"
    path.write_bytes(original)
    config = _config()
    converter = DoclingLocalConverter(config)

    if slice_size is None:
        docling_doc = await converter.convert_file(path)
    else:
        docling_doc = await convert_pdf_with_splitting(
            converter, path, source_uri=None, slice_size=slice_size
        )

    assert len(docling_doc.pages) == 8
    _assert_renders_like(
        docling_doc, original, config.processing.conversion_options.images_scale
    )


@pytest.mark.parametrize("widget", [True, False], ids=["form field", "annotation"])
async def test_annotations_render_as_docling_draws_them(tmp_path, widget):
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    original = _annotated_pdf(widget)
    path = tmp_path / "annotated.pdf"
    path.write_bytes(original)
    config = _config()
    config.processing.conversion_options.do_ocr = False

    docling_doc = await DoclingLocalConverter(config).convert_file(path)

    [rendered] = render_original_pages(original, [1], 2.0).values()
    raster = _raster(docling_doc.pages[1])
    difference = np.abs(
        np.asarray(raster.convert("L"), dtype=np.float32)
        - np.asarray(rendered.convert("L"), dtype=np.float32)
    ).mean()
    assert difference < 1.0


@pytest.mark.parametrize("fmt", ["PNG", "TIFF"])
async def test_image_pages_render_in_docling_frame(tmp_path, fmt):
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    source = pdfium.PdfDocument(str(DOCLAYNET))
    frames = [source[i].render(scale=1.0).to_pil() for i in range(2)]
    buffer = BytesIO()
    frames[0].save(buffer, format=fmt, save_all=fmt == "TIFF", append_images=frames[1:])
    original = buffer.getvalue()
    path = tmp_path / f"scan.{fmt.lower()}"
    path.write_bytes(original)

    docling_doc = await DoclingLocalConverter(_config()).convert_file(path)

    assert len(docling_doc.pages) == (len(frames) if fmt == "TIFF" else 1)
    rendered = render_original_pages(original, sorted(docling_doc.pages), 2.0)
    for page_no, page in docling_doc.pages.items():
        image, raster = rendered[page_no], _raster(page)
        assert (page.size.width, page.size.height) == image.size
        assert _best_shift(raster, image.resize(raster.size)) == (0, 0)


async def test_an_exif_rotated_image_renders_in_docling_frame(tmp_path):
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    scan = pdfium.PdfDocument(str(DOCLAYNET))[0].render(scale=1.0).to_pil()
    exif = scan.getexif()
    exif[0x0112] = 6
    buffer = BytesIO()
    scan.save(buffer, format="JPEG", exif=exif)
    original = buffer.getvalue()
    path = tmp_path / "rotated.jpg"
    path.write_bytes(original)

    docling_doc = await DoclingLocalConverter(_config()).convert_file(path)

    [image] = render_original_pages(original, [1], 2.0).values()
    raster = _raster(docling_doc.pages[1])
    assert image.size == (scan.height, scan.width)
    assert _best_shift(raster, image.resize(raster.size)) == (0, 0)
