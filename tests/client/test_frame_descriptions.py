import logging
from pathlib import Path

from docling_core.types.doc import DocumentOrigin, ImageRef
from PIL import Image
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from haiku.rag.client.processing import convert
from haiku.rag.config import AppConfig
from haiku.rag.providers.picture_description import undescribed_pictures
from tests.conftest import capture_logs
from tests.documents import timed_docling_doc

DESCRIPTION_CONFIG = {
    "embeddings": {"model": {"multimodal": True}},
    "processing": {
        "pictures": "description",
        "media": {"asr": {"base_url": "http://localhost:11470/v1", "model": "whisper"}},
    },
}


def _video():
    doc = timed_docling_doc()
    doc.origin = DocumentOrigin(
        filename="talk.mp4", mimetype="video/mp4", binary_hash="ab" * 32
    )
    return doc


def _no_vlm(monkeypatch) -> None:
    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("a frame was sent to the description model")

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.get_model",
        lambda model_config, app_config: FunctionModel(reply),
    )


async def test_frames_of_a_recording_are_never_described(monkeypatch, tmp_path):
    _no_vlm(monkeypatch)

    async def document(self, path: Path):
        return _video()

    monkeypatch.setattr("haiku.rag.converters.media.MediaConverter._document", document)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")

    with capture_logs(logging.getLogger("haiku.rag"), logging.WARNING) as records:
        doc = await convert(AppConfig.model_validate(DESCRIPTION_CONFIG), video)

    assert [p.meta for p in doc.pictures] == [None, None]
    assert records == []


def test_a_recording_has_no_pictures_to_describe():
    document = timed_docling_doc()
    document.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (96, 96), "red"), dpi=72)
    )

    assert undescribed_pictures(_video()) == []
    assert len(undescribed_pictures(document)) == 3
