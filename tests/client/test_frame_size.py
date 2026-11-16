import base64
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DoclingDocument,
    ImageRef,
    Size,
    TrackSource,
)
from PIL import Image
from pydantic import AnyUrl
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from haiku.rag.client.processing import convert
from haiku.rag.config import AppConfig


def _config(frames: dict | None = None, **processing) -> AppConfig:
    media: dict = {"asr": "whisper_turbo"}
    if frames is not None:
        media["frames"] = frames
    return AppConfig.model_validate({"processing": {"media": media, **processing}})


class _VideoConverter:
    supported_extensions = [".mp4"]

    async def convert_file(self, path: Path, source_uri: str | None = None):
        doc = DoclingDocument(name="talk")
        for start, width in ((0.0, 1920), (5.0, 640)):
            doc.add_picture(
                image=ImageRef.from_pil(
                    Image.new("RGB", (width, width * 9 // 16), "blue"), dpi=72
                ),
                content_layer=ContentLayer.BODY,
                source=TrackSource(start_time=start, end_time=start + 0.001),
            )
        doc.add_picture(
            image=ImageRef.from_pil(Image.new("RGB", (1920, 1080), "red"), dpi=72)
        )
        return doc


async def _convert(monkeypatch, tmp_path, config: AppConfig) -> DoclingDocument:
    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter", lambda config: _VideoConverter()
    )
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    return await convert(config, video)


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


async def test_frames_are_stored_as_jpeg_no_wider_than_max_width(monkeypatch, tmp_path):
    doc = await _convert(monkeypatch, tmp_path, _config())

    wide, narrow, figure = (_stored(p) for p in doc.pictures)
    assert wide == ("JPEG", "image/jpeg", (1280, 720), (1280, 720))
    assert narrow == ("JPEG", "image/jpeg", (640, 360), (640, 360))
    assert figure == ("PNG", "image/png", (1920, 1080), (1920, 1080))


async def test_max_width_follows_config(monkeypatch, tmp_path):
    config = _config({"max_width": 320})

    doc = await _convert(monkeypatch, tmp_path, config)

    assert [_stored(p)[2] for p in doc.pictures] == [
        (320, 180),
        (320, 180),
        (1920, 1080),
    ]


async def test_no_max_width_keeps_docling_frames(monkeypatch, tmp_path):
    config = _config({"max_width": None})

    doc = await _convert(monkeypatch, tmp_path, config)

    assert [_stored(p)[:3] for p in doc.pictures] == [
        ("PNG", "image/png", (1920, 1080)),
        ("PNG", "image/png", (640, 360)),
        ("PNG", "image/png", (1920, 1080)),
    ]


async def test_frames_are_described_from_the_stored_jpeg(monkeypatch, tmp_path):
    sent: list[tuple[str, tuple[int, int]]] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(
                    part.content, str
                ):
                    for item in part.content:
                        if isinstance(item, BinaryContent):
                            with Image.open(BytesIO(item.data)) as image:
                                sent.append((item.media_type, image.size))
        return ModelResponse(parts=[TextPart("A blue slide.")])

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.get_model",
        lambda model_config, app_config: FunctionModel(reply),
    )
    config = _config(pictures="description")

    doc = await _convert(monkeypatch, tmp_path, config)

    assert sent == [("image/jpeg", (1280, 720)), ("image/jpeg", (640, 360))]
    assert [
        p.meta.description.text if p.meta and p.meta.description else None
        for p in doc.pictures
    ] == ["A blue slide.", "A blue slide.", None]


async def test_a_frame_without_inline_image_data_is_left_alone(monkeypatch, tmp_path):
    reference = ImageRef(
        mimetype="image/png",
        dpi=72,
        size=Size(width=1920, height=1080),
        uri=AnyUrl("file:///frames/0001.png"),
    )

    class _ReferencedFrames(_VideoConverter):
        async def convert_file(self, path: Path, source_uri: str | None = None):
            doc = DoclingDocument(name="talk")
            for start, image in ((0.0, reference), (5.0, None)):
                doc.add_picture(
                    image=image,
                    content_layer=ContentLayer.BODY,
                    source=TrackSource(start_time=start, end_time=start + 0.001),
                )
            return doc

    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter",
        lambda config: _ReferencedFrames(),
    )
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")

    doc = await convert(_config(), video)

    assert [p.image for p in doc.pictures] == [reference, None]


def test_a_one_pixel_width_keeps_frames_at_least_one_pixel_high():
    from haiku.rag.client.processing import _shrink_frames

    doc = DoclingDocument(name="talk")
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (1920, 800), "blue"), dpi=72),
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=0.0, end_time=0.001),
    )

    _shrink_frames(_config({"max_width": 1}), doc)

    assert _stored(doc.pictures[0])[2] == (1, 1)


async def test_stored_frame_items_are_the_jpeg(monkeypatch, tmp_path):
    from haiku.rag.store.models.document_item import extract_items

    doc = await _convert(monkeypatch, tmp_path, _config())

    stored = [
        i.picture_data for i in extract_items("doc-1", doc) if i.label == "picture"
    ]
    sizes = []
    for data in stored:
        assert data is not None
        with Image.open(BytesIO(data)) as image:
            sizes.append((image.format, image.size))
    assert sizes == [("JPEG", (1280, 720)), ("JPEG", (640, 360)), ("PNG", (1920, 1080))]


def test_a_frame_described_before_shrinking_keeps_its_description():
    from haiku.rag.client.processing import _shrink_frames
    from haiku.rag.providers.picture_description import set_picture_descriptions

    doc = DoclingDocument(name="talk")
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (1920, 1080), "blue"), dpi=72),
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=0.0, end_time=0.001),
    )
    set_picture_descriptions(doc, {doc.pictures[0].self_ref: "A blue slide."})

    _shrink_frames(_config(), doc)

    meta = doc.pictures[0].meta
    assert meta is not None and meta.description is not None
    assert meta.description.text == "A blue slide."
    assert _stored(doc.pictures[0])[2] == (1280, 720)
