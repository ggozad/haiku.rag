import logging
from pathlib import Path

import pytest
from docling_core.types.doc import ImageRef
from PIL import Image
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from haiku.rag.chunkers import get_chunker
from haiku.rag.client.processing import convert
from haiku.rag.config import AppConfig
from haiku.rag.store.models.document_item import extract_items
from tests.conftest import capture_logs
from tests.documents import timed_docling_doc

DESCRIPTION_CONFIG = {"processing": {"pictures": "description"}}


class _VideoConverter:
    supported_extensions = [".mp4"]

    def __init__(self, with_document_picture: bool = False):
        self.with_document_picture = with_document_picture

    async def convert_file(self, path: Path, source_uri: str | None = None):
        doc = timed_docling_doc()
        if self.with_document_picture:
            doc.add_picture(
                image=ImageRef.from_pil(Image.new("RGB", (96, 96), "red"), dpi=72)
            )
        return doc


def _vlm(monkeypatch, replies: list[str | Exception]) -> list[str | Exception]:
    remaining = list(replies)

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answer = remaining.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return ModelResponse(parts=[TextPart(answer)])

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.get_model",
        lambda model_config, app_config: FunctionModel(reply),
    )
    return remaining


async def _convert(
    monkeypatch, tmp_path, config: AppConfig, converter: _VideoConverter
):
    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter", lambda config: converter
    )
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    return await convert(config, video)


def _descriptions(doc) -> list[str | None]:
    return [
        p.meta.description.text if p.meta and p.meta.description else None
        for p in doc.pictures
    ]


async def test_frames_are_described_at_ingest(monkeypatch, tmp_path):
    remaining = _vlm(monkeypatch, ["A blue slide.", "A green slide.", "A red square."])
    config = AppConfig.model_validate(DESCRIPTION_CONFIG)

    doc = await _convert(
        monkeypatch, tmp_path, config, _VideoConverter(with_document_picture=True)
    )

    assert _descriptions(doc) == ["A blue slide.", "A green slide.", None]
    assert remaining == ["A red square."]
    items = {i.self_ref: i.text for i in extract_items("doc-1", doc)}
    assert "A blue slide." in items[doc.pictures[0].self_ref]
    chunks = await get_chunker(config).chunk(doc)
    assert "A green slide." in " ".join(c.content for c in chunks)


async def test_a_failed_frame_stays_undescribed(monkeypatch, tmp_path):
    _vlm(monkeypatch, ["A blue slide.", RuntimeError("VLM down")])
    config = AppConfig.model_validate(DESCRIPTION_CONFIG)

    with capture_logs(logging.getLogger("haiku.rag"), logging.WARNING) as records:
        doc = await _convert(monkeypatch, tmp_path, config, _VideoConverter())

    assert _descriptions(doc) == ["A blue slide.", None]
    assert [r.getMessage() for r in records] == [
        f"VLM call failed for {doc.pictures[1].self_ref}: VLM down"
    ]


@pytest.mark.parametrize(
    "replies", [["   ", "\n"], [RuntimeError("VLM down"), RuntimeError("VLM down")]]
)
async def test_no_described_frame_warns_once(monkeypatch, tmp_path, replies):
    _vlm(monkeypatch, replies)
    config = AppConfig.model_validate(DESCRIPTION_CONFIG)

    with capture_logs(logging.getLogger("haiku.rag"), logging.WARNING) as records:
        doc = await _convert(monkeypatch, tmp_path, config, _VideoConverter())

    assert _descriptions(doc) == [None, None]
    summaries = [r for r in records if "no descriptions came back" in r.getMessage()]
    assert len(summaries) == 1
    assert len(records) == 1 + sum(isinstance(r, Exception) for r in replies)


async def test_frames_are_not_described_without_description_mode(monkeypatch, tmp_path):
    remaining = _vlm(monkeypatch, ["unused"])

    doc = await _convert(monkeypatch, tmp_path, AppConfig(), _VideoConverter())

    assert _descriptions(doc) == [None, None]
    assert remaining == ["unused"]


@pytest.mark.vcr()
async def test_described_frames_are_stored_and_searchable(
    monkeypatch, tmp_path, temp_db_path
):
    from haiku.rag.client import HaikuRAG

    _vlm(monkeypatch, ["A blue slide of the flight plan.", RuntimeError("VLM down")])
    converter = _VideoConverter()
    for module in ("processing", "documents"):
        monkeypatch.setattr(
            f"haiku.rag.client.{module}.get_converter", lambda config: converter
        )
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")

    with capture_logs(logging.getLogger("haiku.rag"), logging.WARNING) as records:
        async with HaikuRAG(
            temp_db_path,
            config=AppConfig.model_validate(DESCRIPTION_CONFIG),
            create=True,
        ) as rag:
            doc = await rag.create_document_from_source(video)
            assert not isinstance(doc, list) and doc.id is not None
            items = await rag.document_item_repository.get_all_items(doc.id)
            results = await rag.search("flight plan slide", search_type="fts")

    frames = [i.text for i in items if i.label == "picture"]
    assert frames == ["A blue slide of the flight plan.", ""]
    assert any("A blue slide of the flight plan." in r.content for r in results)
    assert [r.getMessage() for r in records] == [
        "VLM call failed for #/pictures/1: VLM down"
    ]
