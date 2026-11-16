from io import BytesIO

import pypdfium2 as pdfium
import pytest
from PIL import Image

from haiku.rag.utils.images import (
    decode_picture,
    original_layout,
    render_original_pages,
)


def _pdf(*sizes: tuple[float, float]) -> bytes:
    pdf = pdfium.PdfDocument.new()
    for width, height in sizes:
        pdf.new_page(width, height)
    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def _image(fmt: str, *colors: str, **params) -> bytes:
    frames = [Image.new("RGB", (120, 80), color) for color in colors]
    buffer = BytesIO()
    frames[0].save(
        buffer, format=fmt, save_all=True, append_images=frames[1:], **params
    )
    return buffer.getvalue()


def test_renders_the_requested_pdf_pages_at_scale():
    pages = render_original_pages(_pdf((200, 100), (300, 400), (50, 50)), [2, 3], 2.0)

    assert {
        n: (page.image.size, page.width, page.height) for n, page in pages.items()
    } == {2: ((600, 800), 300, 400), 3: ((100, 100), 50, 50)}


def test_a_rotated_pdf_page_has_its_displayed_size():
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 100).set_rotation(90)
    buffer = BytesIO()
    pdf.save(buffer)

    [page] = render_original_pages(buffer.getvalue(), [1], 1.0).values()

    assert (page.image.size, page.width, page.height) == ((100, 200), 100, 200)


def test_an_image_is_one_page_at_its_own_size():
    pages = render_original_pages(_image("PNG", "red"), [1], 2.0)

    assert {
        n: (page.image.size, page.width, page.height, page.image.getpixel((0, 0)))
        for n, page in pages.items()
    } == {1: ((120, 80), 120, 80, (255, 0, 0))}


@pytest.mark.parametrize(
    ("dpi", "size"),
    [((300, 300), (28.8, 19.2)), ((150, 300), (57.6, 19.2)), ((1, 1), (120, 80))],
)
def test_an_image_page_is_sized_in_points_from_its_dpi(dpi, size):
    [page] = render_original_pages(_image("TIFF", "red", dpi=dpi), [1], 2.0).values()

    assert page.image.size == (120, 80)
    assert (page.width, page.height) == pytest.approx(size, abs=0.01)


def test_a_multi_frame_tiff_has_a_page_per_frame():
    pages = render_original_pages(_image("TIFF", "red", "blue"), [1, 2], 2.0)

    assert {n: page.image.getpixel((0, 0)) for n, page in pages.items()} == {
        1: (255, 0, 0),
        2: (0, 0, 255),
    }


def test_pages_past_the_end_are_left_out():
    assert render_original_pages(_pdf((100, 100)), [2], 1.0) == {}
    assert render_original_pages(_image("PNG", "red"), [2], 1.0) == {}


def test_an_image_follows_its_exif_orientation():
    image = Image.new("RGB", (120, 80), "red")
    exif = image.getexif()
    exif[0x0112] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif, dpi=(144, 72))

    [page] = render_original_pages(buffer.getvalue(), [1], 2.0).values()

    assert page.image.size == (80, 120)
    assert (page.width, page.height) == pytest.approx((80, 60), abs=0.01)


def test_a_square_image_turned_by_exif_keeps_its_dpi_axes():
    image = Image.new("RGB", (100, 100), "red")
    exif = image.getexif()
    exif[0x0112] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif, dpi=(144, 72))

    [page] = render_original_pages(buffer.getvalue(), [1], 2.0).values()

    assert (page.width, page.height) == pytest.approx((50, 100), abs=0.01)


def test_transparency_renders_on_white():
    buffer = BytesIO()
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(buffer, format="PNG")

    [page] = render_original_pages(buffer.getvalue(), [1], 2.0).values()

    assert (page.image.mode, page.image.getpixel((0, 0))) == ("RGB", (255, 255, 255))


def _rotated_pdf() -> bytes:
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 100)
    pdf.new_page(300, 400).set_rotation(90)
    cropped = pdf.new_page(300, 400)
    cropped.set_cropbox(10, 20, 210, 320)
    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def _exif_jpeg() -> bytes:
    image = Image.new("RGB", (120, 80), "red")
    exif = image.getexif()
    exif[0x0112] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif, dpi=(144, 72))
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("original", "pdf"),
    [
        (_rotated_pdf(), True),
        (_image("TIFF", "red", "blue", dpi=(150, 300)), False),
        (_exif_jpeg(), False),
    ],
    ids=["pdf", "tiff", "exif"],
)
def test_the_layout_gives_every_page_the_size_rendering_gives(original, pdf):
    layout = original_layout(original)

    rendered = render_original_pages(original, list(range(1, 10)), 1.0)
    assert layout.pdf is pdf
    assert layout.sizes == {
        n: pytest.approx((page.width, page.height)) for n, page in rendered.items()
    }


@pytest.mark.parametrize(
    ("fmt", "media_type"), [("PNG", "image/png"), ("JPEG", "image/jpeg")]
)
def test_a_decoded_picture_carries_its_own_media_type(fmt, media_type):
    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format=fmt)

    picture = decode_picture(buf.getvalue(), "#/pictures/0")

    assert picture is not None
    assert picture.media_type == media_type
