import base64
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from haiku.rag.client.processing import convert
from haiku.rag.config import AppConfig
from haiku.rag.converters import media
from tests.platform import needs_ffmpeg

pytestmark = needs_ffmpeg

VIDEO = Path(__file__).parents[1] / "data" / "speech.mp4"


@pytest.fixture(autouse=True)
def silent_transcription(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"segments": []})
    )
    monkeypatch.setattr(
        media,
        "pooled_client",
        lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout),
    )


def _config(frames: dict | None = None, **processing) -> AppConfig:
    media_config: dict = {
        "asr": {"base_url": "http://localhost:11470/v1", "model": "whisper"}
    }
    if frames is not None:
        media_config["frames"] = frames
    return AppConfig.model_validate(
        {
            "embeddings": {"model": {"multimodal": True}},
            "processing": {"media": media_config, **processing},
        }
    )


def _stored(picture) -> tuple[str, str | None, tuple[int, int], tuple[float, float]]:
    """The format, declared mimetype, decoded size and declared size of a picture."""
    head, encoded = str(picture.image.uri).split(",", 1)
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        fmt, size = image.format or "", image.size
    assert head == f"data:{picture.image.mimetype};base64"
    return (
        fmt,
        picture.image.mimetype,
        size,
        (picture.image.size.width, picture.image.size.height),
    )


async def test_frames_are_jpeg_at_the_video_s_width_below_max_width():
    doc = await convert(_config(), VIDEO)

    assert [_stored(p) for p in doc.pictures] == [
        ("JPEG", "image/jpeg", (320, 240), (320, 240))
    ] * 2


@pytest.mark.parametrize(
    ("max_width", "size"), [(160, (160, 120)), (1, (1, 1)), (None, (320, 240))]
)
async def test_frames_are_no_wider_than_max_width(max_width, size):
    doc = await convert(_config({"max_width": max_width}), VIDEO)

    assert [_stored(p)[2] for p in doc.pictures] == [size, size]


async def test_stored_frame_items_are_the_jpeg():
    from haiku.rag.store.models.document_item import extract_items

    doc = await convert(_config({"max_width": 160}), VIDEO)

    stored = [
        i.picture_data for i in extract_items("doc-1", doc) if i.label == "picture"
    ]
    sizes = []
    for data in stored:
        assert data is not None
        with Image.open(BytesIO(data)) as image:
            sizes.append((image.format, image.size))
    assert sizes == [("JPEG", (160, 120))] * 2
