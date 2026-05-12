"""Tests for haiku.rag.client.processing helpers."""

import logging
from pathlib import Path

import pytest

from haiku.rag.client.processing import _warn_if_descriptions_missing, convert
from haiku.rag.config import AppConfig


def _doc_with_pictures(*, with_descriptions: bool):
    """Build a tiny DoclingDocument carrying one PictureItem.

    When ``with_descriptions=True`` the picture's ``meta.description.text``
    is populated, simulating a successful VLM call. Otherwise it's left
    empty, simulating a silent VLM failure.
    """
    from docling_core.types.doc.document import (
        DescriptionMetaField,
        DoclingDocument,
        PictureItem,
        PictureMeta,
    )
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="test")
    pic = PictureItem(
        self_ref="#/pictures/0",
        label=DocItemLabel.PICTURE,
    )
    if with_descriptions:
        pic.meta = PictureMeta(
            description=DescriptionMetaField(text="A red square."),
        )
    doc.pictures.append(pic)
    return doc


def _doc_without_pictures():
    from docling_core.types.doc.document import DoclingDocument

    return DoclingDocument(name="test")


@pytest.fixture
def caplog_warnings(caplog):
    """Capture WARNING-level records from the processing logger."""
    caplog.set_level(logging.WARNING, logger="haiku.rag.client.processing")
    # The haiku.rag parent logger sets propagate=False after get_logger() runs,
    # which can break caplog under xdist when other tests have already
    # configured logging. Attach directly to the module logger.
    from haiku.rag.client.processing import logger as proc_logger

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    proc_logger.addHandler(handler)
    yield records
    proc_logger.removeHandler(handler)


def test_no_warning_when_picture_description_disabled(caplog_warnings):
    """Default config has picture_description.enabled=False — even a
    document full of pictures with no descriptions should not warn."""
    config = AppConfig()
    assert config.processing.conversion_options.picture_description.enabled is False
    doc = _doc_with_pictures(with_descriptions=False)

    _warn_if_descriptions_missing(config, doc, "fake.pdf")

    assert caplog_warnings == []


def test_no_warning_when_doc_has_no_pictures(caplog_warnings):
    """A picture-less document under enabled=True shouldn't warn — there
    was simply nothing to describe."""
    config = AppConfig()
    config.processing.conversion_options.picture_description.enabled = True
    doc = _doc_without_pictures()

    _warn_if_descriptions_missing(config, doc, "no-pictures.txt")

    assert caplog_warnings == []


def test_warns_when_pictures_present_but_no_descriptions(caplog_warnings):
    """VLM was requested via ``picture_description.enabled = True``, the
    doc has pictures, but the converter returned zero descriptions
    (docling-serve swallows VLM errors). Warn loudly so the user can fix
    their VLM config before a long ingest."""
    config = AppConfig()
    config.processing.conversion_options.picture_description.enabled = True
    config.processing.conversion_options.picture_description.model.name = "qwen3.6"
    config.processing.conversion_options.picture_description.model.base_url = (
        "http://host.docker.internal:11434"
    )
    doc = _doc_with_pictures(with_descriptions=False)

    _warn_if_descriptions_missing(config, doc, "doclaynet.pdf")

    assert len(caplog_warnings) == 1
    msg = caplog_warnings[0].getMessage()
    assert "doclaynet.pdf" in msg
    assert "1 pictures" in msg
    assert "0 described" in msg
    assert "qwen3.6" in msg
    assert "host.docker.internal" in msg


def test_no_warning_when_at_least_one_description_came_back(caplog_warnings):
    """Partial coverage (some pictures described, some not) is acceptable
    and does not warn — the user might have area-threshold filtering or
    classification gating."""
    config = AppConfig()
    config.processing.conversion_options.picture_description.enabled = True
    doc = _doc_with_pictures(with_descriptions=True)

    _warn_if_descriptions_missing(config, doc, "fine.pdf")

    assert caplog_warnings == []


@pytest.mark.asyncio
async def test_convert_emits_warning_via_chokepoint(
    monkeypatch, tmp_path, caplog_warnings
):
    """End-to-end: ``convert(...)`` runs the description-missing check
    after the converter returns, so a VLM error swallowed inside
    docling-serve still surfaces as a warning at the haiku.rag layer
    regardless of which converter (local vs serve) ran."""
    from haiku.rag.converters.base import DocumentConverter

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    class StubConverter(DocumentConverter):
        @property
        def supported_extensions(self) -> list[str]:
            return [".pdf"]

        async def convert_file(self, path: Path, source_uri: str | None = None):
            return _doc_with_pictures(with_descriptions=False)

        async def convert_text(
            self,
            text: str,
            name: str = "content.md",
            format: str = "md",
            source_uri: str | None = None,
        ):
            return _doc_without_pictures()

    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter", lambda config: StubConverter()
    )

    config = AppConfig()
    config.processing.conversion_options.picture_description.enabled = True

    await convert(config, pdf)

    assert any(
        "0 described" in r.getMessage() and "fake.pdf" in r.getMessage()
        for r in caplog_warnings
    )


@pytest.mark.asyncio
async def test_convert_text_path_also_warns(monkeypatch, caplog_warnings):
    """Raw text input (HTML, markdown) can still produce pictures via
    docling, so the description-missing check must run on the
    convert_text branch too — otherwise an HTML-with-images source
    would never trigger the warning even when picture_description is
    enabled and the VLM didn't actually run."""
    from haiku.rag.converters.base import DocumentConverter

    class StubConverter(DocumentConverter):
        @property
        def supported_extensions(self) -> list[str]:
            return [".html"]

        async def convert_file(self, path: Path, source_uri: str | None = None):
            return _doc_without_pictures()

        async def convert_text(
            self,
            text: str,
            name: str = "content.md",
            format: str = "md",
            source_uri: str | None = None,
        ):
            return _doc_with_pictures(with_descriptions=False)

    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter", lambda config: StubConverter()
    )

    config = AppConfig()
    config.processing.conversion_options.picture_description.enabled = True

    # No URL scheme, no Path → drops into the convert_text branch.
    await convert(config, "<html><img src='...'/></html>")

    assert any("0 described" in r.getMessage() for r in caplog_warnings)
