from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.citation import resolve_citations


def _result(
    chunk_id: str,
    chunk_ids: list[str] | None = None,
    document_meta: dict | None = None,
    chunk_meta: dict | None = None,
) -> SearchResult:
    return SearchResult(
        content="content",
        score=0.9,
        chunk_id=chunk_id,
        chunk_ids=chunk_ids or [],
        document_id="doc-1",
        document_uri="test://doc",
        document_meta=document_meta or {},
        chunk_meta=chunk_meta or {},
    )


def test_resolve_citations_copies_merged_chunk_ids():
    result = _result("c1", chunk_ids=["c1", "c2"])
    citations = resolve_citations(["c1"], [result])
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"
    assert citations[0].chunk_ids == ["c1", "c2"]


def test_resolve_citations_falls_back_to_chunk_id():
    result = _result("c1")
    citations = resolve_citations(["c1"], [result])
    assert len(citations) == 1
    assert citations[0].chunk_ids == ["c1"]


def test_resolve_citations_strips_brackets():
    result = _result("c1")
    citations = resolve_citations(["[c1]"], [result])
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"


def test_resolve_citations_skips_unknown_ids():
    result = _result("c1")
    citations = resolve_citations(["c1", "missing"], [result])
    assert len(citations) == 1


def test_resolve_citations_copies_document_meta():
    result = _result(
        "c1", document_meta={"source_url": "https://example.org/report/view"}
    )
    citations = resolve_citations(["c1"], [result])
    assert citations[0].document_meta == {
        "source_url": "https://example.org/report/view"
    }


def test_resolve_citations_copies_chunk_meta():
    result = _result("c1", chunk_meta={"para_no": "12", "speaker": "MR SMITH"})
    citations = resolve_citations(["c1"], [result])
    assert citations[0].chunk_meta == {"para_no": "12", "speaker": "MR SMITH"}
