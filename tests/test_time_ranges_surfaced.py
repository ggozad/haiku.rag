import pytest

from haiku.rag.context import expand_with_items
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.store.models.citation import resolve_citations
from haiku.rag.store.models.document_item import extract_items
from haiku.rag.utils.formatting import format_time_range
from tests.documents import timed_docling_doc

SENTENCES = [
    "The oxygen tank failed fifty five hours into the flight.",
    "The crew moved into the lunar module.",
    "Engineers improvised a carbon dioxide filter.",
]


@pytest.mark.parametrize(
    ("time_range", "rendered"),
    [
        ((4.48, 7.6), "00:00:04-00:00:07"),
        ((725.0, 3725.9), "00:12:05-01:02:05"),
        ((10.0, 10.001), "00:00:10"),
        ((0.9, 1.4), "00:00:00-00:00:01"),
    ],
)
def test_format_time_range(time_range, rendered):
    assert format_time_range(time_range) == rendered


def _result(ref: str, content: str, time_range, score: float = 1.0) -> SearchResult:
    return SearchResult(
        content=content,
        score=score,
        chunk_id=ref,
        document_id="doc-1",
        doc_item_refs=[ref],
        time_range=time_range,
    )


def _expand(results, max_chars):
    items = extract_items("doc-1", timed_docling_doc())
    positions = {item.self_ref: item.position for item in items}
    return expand_with_items(
        results,
        max_chars,
        {ref: positions[ref] for r in results for ref in r.doc_item_refs},
        items,
    )


def test_search_result_takes_the_chunk_time_range():
    chunk = Chunk(
        id="c1",
        content="x",
        metadata={"doc_item_refs": ["#/texts/1"], "time_range": [4.48, 7.6]},
    )

    assert SearchResult.from_chunk(chunk, score=1.0).time_range == (4.48, 7.6)


def test_expansion_spans_the_items_it_returns():
    [expanded] = _expand([_result("#/texts/1", SENTENCES[1], (4.48, 7.6))], 10_000)

    assert expanded.time_range == (0.0, 12.12)


def test_a_clipped_expansion_spans_the_surviving_items():
    line = f"[00:00:04] {SENTENCES[1]}"
    [expanded] = _expand([_result("#/texts/1", SENTENCES[1], (4.48, 7.6))], len(line))

    assert expanded.content == line
    assert expanded.time_range == (4.48, 7.6)


def test_a_recording_never_falls_back_to_a_longer_chunk():
    longer = SENTENCES[1] + " " + "padding " * 200
    [expanded] = _expand([_result("#/texts/1", longer, (4.48, 7.6))], 100_000)

    assert expanded.content.splitlines() == [
        f"[00:00:00] {SENTENCES[0]}",
        f"[00:00:04] {SENTENCES[1]}",
        f"[00:00:07] {SENTENCES[2]}",
    ]
    assert expanded.time_range == (0.0, 12.12)


def test_merged_results_span_their_items():
    results = [
        _result("#/texts/0", SENTENCES[0], (0.0, 4.16)),
        _result("#/texts/1", SENTENCES[1], (4.48, 7.6)),
    ]

    lines = [f"[00:00:00] {SENTENCES[0]}", f"[00:00:04] {SENTENCES[1]}"]
    [expanded] = _expand(results, len("\n".join(lines)))

    assert expanded.content == "\n".join(lines)
    assert expanded.time_range == (0.0, 7.6)


def test_a_frame_expands_to_the_transcript_around_it():
    [expanded] = _expand([_result("#/pictures/1", "", (10.0, 10.001))], 10_000)

    assert "#/pictures/1" in expanded.doc_item_refs
    assert expanded.time_range == (0.0, 12.12)


def test_format_for_agent_shows_the_time():
    timed = _result("#/texts/1", SENTENCES[1], (4.48, 7.6))
    untimed = _result("#/texts/1", SENTENCES[1], None)

    assert "Time: 00:00:04-00:00:07" in timed.format_for_agent(rank=1)
    assert "Time:" not in untimed.format_for_agent(rank=1)


def test_citations_carry_the_time_range():
    result = _result("#/texts/1", SENTENCES[1], (4.48, 7.6))
    result.document_uri = "talk.mp4"

    [citation] = resolve_citations(["#/texts/1"], [result])

    assert citation.time_range == (4.48, 7.6)


def _citation(time_range):
    from haiku.rag.store.models.citation import Citation

    return Citation(
        document_id="doc-1",
        chunk_id="c1",
        document_uri="talk.mp4",
        document_title="Talk",
        content=SENTENCES[1],
        time_range=time_range,
    )


async def test_cli_citations_show_the_time():
    from rich.console import Console

    from haiku.rag.utils.formatting import format_citations_rich

    console = Console(record=True, width=200)
    for r in await format_citations_rich([_citation((4.48, 7.6))]):
        console.print(r)

    assert "[1] Talk (talk.mp4) — 00:00:04-00:00:07" in console.export_text()


def test_the_compaction_capsule_shows_the_time():
    from haiku.rag.capabilities.compaction import _Entry

    entry = _Entry(
        capability="rag", chunk_id="c1", question=0, citation=_citation((4.48, 7.6))
    )
    untimed = _Entry(
        capability="rag", chunk_id="c1", question=0, citation=_citation(None)
    )

    assert entry.render().splitlines()[:2] == [
        '[c1] Source: "Talk" (talk.mp4)',
        "Time: 00:00:04-00:00:07",
    ]
    assert "Time:" not in untimed.render()


async def test_sandbox_items_carry_their_times(temp_db_path):
    import json

    from haiku.rag.client import HaikuRAG
    from haiku.rag.config.models import AppConfig
    from haiku.rag.sandbox import AnalysisContext, Sandbox
    from tests.sandbox.test_sandbox_toc import _empty_doc, _read_vfs_text

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc_id = await _empty_doc(client, uri="talk.mp4", title="Talk")
        await client.document_item_repository.create_items(
            doc_id, extract_items(doc_id, timed_docling_doc())
        )

    sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
    raw = await _read_vfs_text(sandbox, f"/documents/{doc_id}/items.jsonl")
    items = [json.loads(line) for line in raw.splitlines()]

    assert [(i["start_time"], i["end_time"]) for i in items][:2] == [
        (0.0, 0.001),
        (0.0, 4.16),
    ]


def test_sandbox_rows_carry_times_only_for_recordings():
    from haiku.rag.sandbox.sandbox import _item_row, _search_row
    from tests.documents import docling_doc

    [untimed_item] = extract_items("doc-1", docling_doc("plain", "No timing."))
    timed_item = extract_items("doc-1", timed_docling_doc())[1]

    assert "start_time" not in _item_row(untimed_item, [])
    assert (
        _item_row(timed_item, [])["start_time"],
        _item_row(timed_item, [])["end_time"],
    ) == (0.0, 4.16)
    assert "time_range" not in _search_row(_result("#/texts/1", "x", None))
    assert _search_row(_result("#/texts/1", "x", (4.48, 7.6)))["time_range"] == (
        4.48,
        7.6,
    )


def _recording(*events: tuple[str, float]):
    """A recording of text items and frames, `("frame", t)` for a frame."""
    from docling_core.types.doc import ContentLayer, DocItemLabel, ImageRef, TrackSource
    from docling_core.types.doc.document import DoclingDocument
    from PIL import Image

    doc = DoclingDocument(name="recording")
    for text, at in events:
        source = TrackSource(start_time=at, end_time=at + 4.0)
        if text == "frame":
            doc.add_picture(
                image=ImageRef.from_pil(Image.new("RGB", (8, 8)), dpi=72),
                content_layer=ContentLayer.BODY,
                source=source,
            )
        else:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                content_layer=ContentLayer.BODY,
                source=source,
            )
    return extract_items("doc-1", doc)


def _expand_items(results, max_chars, items):
    positions = {item.self_ref: item.position for item in items}
    return expand_with_items(
        results,
        max_chars,
        {ref: positions[ref] for r in results for ref in r.doc_item_refs},
        items,
    )


def test_a_shrunk_clip_stays_on_the_hit_when_part_of_it_repeats_earlier():
    from haiku.rag.store.models.document_item import DocumentItem

    hit = "S" * 36 + "M" * 128 + "E" * 36
    item = DocumentItem(
        document_id="doc-1",
        position=0,
        self_ref="#/texts/0",
        label="text",
        text="M" * 128 + "Z" * 100 + hit,
        start_time=0.0,
        end_time=30.0,
    )

    [expanded] = _expand_items([_result("#/texts/0", hit, (0.0, 30.0))], 200, [item])

    assert len(expanded.content) <= 200
    assert "Z" not in expanded.content
    assert "SSS" in expanded.content and "EEE" in expanded.content


def test_a_frame_clipped_out_of_the_text_loses_its_image_too():
    items = _recording(
        ("frame", 0.0),
        ("A sentence said before the hit, long enough to clip.", 1.0),
        ("The hit.", 5.0),
    )
    hit = _result("#/texts/1", "The hit.", (5.0, 9.0))
    hit.image_data = {"#/pictures/0": "aGk="}
    hit.picture_times = {"#/pictures/0": 0.0}

    [expanded] = _expand_items([hit], len("[00:00:05] The hit."), items)

    assert expanded.content == "[00:00:05] The hit."
    assert not expanded.image_data
    assert not expanded.picture_times


def test_a_frame_marker_longer_than_the_budget_is_left_out():
    items = _recording(("frame", 10.0))
    frame = _result("#/pictures/0", "", (10.0, 10.001))
    frame.image_data = {"#/pictures/0": "aGk="}
    frame.picture_times = {"#/pictures/0": 10.0}

    [expanded] = _expand_items([frame], 10, items)

    assert len(expanded.content) <= 10
    assert not expanded.image_data
