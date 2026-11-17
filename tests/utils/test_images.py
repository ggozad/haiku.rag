from io import BytesIO

import pytest
from PIL import Image

from haiku.rag.utils.images import decode_picture


@pytest.mark.parametrize(
    ("fmt", "media_type"), [("PNG", "image/png"), ("JPEG", "image/jpeg")]
)
def test_a_decoded_picture_carries_its_own_media_type(fmt, media_type):
    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format=fmt)

    picture = decode_picture(buf.getvalue(), "#/pictures/0")

    assert picture is not None
    assert picture.media_type == media_type
