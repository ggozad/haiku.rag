from docling_core.types.doc import DocItemLabel, DoclingDocument, TrackSource

from haiku.rag.converters.base import label_speakers
from tests.documents import timed_docling_doc


def _transcript(*turns: tuple[str | None, str]) -> DoclingDocument:
    doc = DoclingDocument(name="talk")
    for n, (voice, text) in enumerate(turns):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            source=TrackSource(start_time=n, end_time=n + 1, voice=voice),
        )
    return doc


def test_prefixes_every_spoken_item_with_its_speaker():
    doc = _transcript(
        ("speaker-1", "Houston, we have a problem."),
        ("speaker-1", "The main bus is down."),
        ("speaker-2", "Say again."),
    )

    label_speakers(doc)

    assert [t.text for t in doc.texts] == [
        "speaker-1: Houston, we have a problem.",
        "speaker-1: The main bus is down.",
        "speaker-2: Say again.",
    ]


def test_leaves_items_without_a_speaker_unchanged():
    doc = _transcript((None, "Unattributed."), ("speaker-1", "Attributed."))

    label_speakers(doc)

    assert [t.text for t in doc.texts] == ["Unattributed.", "speaker-1: Attributed."]


def test_leaves_a_transcript_without_speakers_unchanged():
    doc = timed_docling_doc()
    before = [t.text for t in doc.texts]

    label_speakers(doc)

    assert [t.text for t in doc.texts] == before


def test_docling_serve_output_gets_speaker_labels():
    import io
    import zipfile

    from haiku.rag.config import AppConfig
    from haiku.rag.converters.docling_serve import DoclingServeConverter

    doc = _transcript(("speaker-1", "Houston, we have a problem."))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("talk.json", doc.model_dump_json())

    parsed = DoclingServeConverter(AppConfig())._parse_zip_to_docling(
        buf.getvalue(), "talk"
    )

    assert parsed.texts[0].text == "speaker-1: Houston, we have a problem."


def test_docling_local_output_gets_speaker_labels(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from haiku.rag.config import AppConfig
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    doc = _transcript(("speaker-1", "Houston, we have a problem."))
    converter = DoclingLocalConverter(AppConfig())
    monkeypatch.setattr(
        converter,
        "_cached_converter",
        lambda: SimpleNamespace(convert=lambda source: SimpleNamespace(document=doc)),
    )

    parsed = converter._sync_convert_timed(Path("talk.pdf"))

    assert parsed.texts[0].text == "speaker-1: Houston, we have a problem."
