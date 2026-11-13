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
    [expanded] = _expand(
        [_result("#/texts/1", SENTENCES[1], (4.48, 7.6))], len(SENTENCES[1])
    )

    assert expanded.content == SENTENCES[1]
    assert expanded.time_range == (4.48, 7.6)


def test_an_expansion_falling_back_to_the_chunk_keeps_its_range():
    longer = SENTENCES[1] + " " + "padding " * 200
    [expanded] = _expand([_result("#/texts/1", longer, (4.48, 7.6))], 100_000)

    assert expanded.content == longer
    assert expanded.time_range == (4.48, 7.6)


def test_a_fallback_keeps_only_the_primary_chunk_range():
    primary = _result(
        "#/texts/1", SENTENCES[1] + " " + "padding " * 200, (0.0, 4.0), 2.0
    )
    other = _result("#/texts/1", SENTENCES[1], (6.0, 10.0), 1.0)

    [expanded] = _expand([primary, other], 100_000)

    assert expanded.content == primary.content
    assert expanded.time_range == (0.0, 4.0)


def test_merged_results_span_their_items():
    results = [
        _result("#/texts/0", SENTENCES[0], (0.0, 4.16)),
        _result("#/texts/1", SENTENCES[1], (4.48, 7.6)),
    ]

    [expanded] = _expand(results, len(SENTENCES[0]) + len(SENTENCES[1]) + 2)

    assert expanded.content == f"{SENTENCES[0]}\n\n{SENTENCES[1]}"
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
