from pathlib import Path

import pytest
from docling_core.types.doc import PictureItem, TrackSource

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import AppConfig
from haiku.rag.converters.base import MEDIA_EXTENSIONS
from haiku.rag.converters.docling_serve import DoclingServeConverter

DATA = Path(__file__).parents[1] / "data"


def _config(base_url: str = "http://localhost:5011", **processing) -> AppConfig:
    return AppConfig.model_validate(
        {
            "processing": {
                "converter": "docling-serve",
                "chunker": "docling-serve",
                **processing,
            },
            "providers": {"docling_serve": {"base_url": base_url}},
        }
    )


def test_media_is_off_by_default():
    converter = DoclingServeConverter(_config())

    assert not set(MEDIA_EXTENSIONS) & set(converter.supported_extensions)
    with pytest.raises(UnsupportedSourceError):
        converter.require_supported(Path("talk.mp3"))


def test_configuring_media_supports_its_extensions():
    converter = DoclingServeConverter(_config(media={}))

    assert set(MEDIA_EXTENSIONS) <= set(converter.supported_extensions)


@pytest.mark.integration
async def test_a_served_recording_is_searchable_with_its_time(
    docling_serve_media_url, temp_db_path
):
    from haiku.rag.client import HaikuRAG

    config = _config(docling_serve_media_url, media={})
    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        video = await rag.convert(DATA / "speech.mp4")
        assert isinstance(video.texts[0].source[0], TrackSource)
        frames = [i for i, _ in video.iterate_items() if isinstance(i, PictureItem)]
        assert frames and all(f.image is not None for f in frames)

        doc = await rag.create_document_from_source(DATA / "speech.mp4")
        assert not isinstance(doc, list)
        results = await rag.search("carbon dioxide filter", limit=3)

    assert results and results[0].document_id == doc.id
    assert results[0].time_range is not None
