from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image

from haiku.rag.utils.images import render_original_pages


def _pdf(*sizes: tuple[float, float]) -> bytes:
    pdf = pdfium.PdfDocument.new()
    for width, height in sizes:
        pdf.new_page(width, height)
    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def _image(fmt: str, *colors: str) -> bytes:
    frames = [Image.new("RGB", (120, 80), color) for color in colors]
    buffer = BytesIO()
    frames[0].save(buffer, format=fmt, save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def test_renders_the_requested_pdf_pages_at_scale():
    pages = render_original_pages(_pdf((200, 100), (300, 400), (50, 50)), [2, 3], 2.0)

    assert {n: page.size for n, page in pages.items()} == {2: (600, 800), 3: (100, 100)}


def test_an_image_is_one_page_at_its_own_size():
    pages = render_original_pages(_image("PNG", "red"), [1], 2.0)

    assert {n: (page.size, page.getpixel((0, 0))) for n, page in pages.items()} == {
        1: ((120, 80), (255, 0, 0))
    }


def test_a_multi_frame_tiff_has_a_page_per_frame():
    pages = render_original_pages(_image("TIFF", "red", "blue"), [1, 2], 2.0)

    assert {n: page.getpixel((0, 0)) for n, page in pages.items()} == {
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
    image.save(buffer, format="JPEG", exif=exif)

    [page] = render_original_pages(buffer.getvalue(), [1], 2.0).values()

    assert page.size == (80, 120)


def test_transparency_renders_on_white():
    buffer = BytesIO()
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(buffer, format="PNG")

    [page] = render_original_pages(buffer.getvalue(), [1], 2.0).values()

    assert (page.mode, page.getpixel((0, 0))) == ("RGB", (255, 255, 255))
