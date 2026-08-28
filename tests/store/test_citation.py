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


def test_a_repeated_chunk_is_cited_from_its_last_occurrence():
    """One chunk is returned by several searches, each expanded against what
    that search found in the same document, so the copies differ in everything
    the window decides. The later entry supplies them."""
    earlier = SearchResult(
        content="narrow window",
        score=0.9,
        chunk_id="c1",
        chunk_ids=["c1"],
        document_id="doc-1",
        document_uri="test://doc",
        doc_item_refs=["#/texts/4"],
        page_numbers=[2],
        headings=["Maintenance"],
    )
    later = SearchResult(
        content="wider window",
        score=0.4,
        chunk_id="c1",
        chunk_ids=["c1", "c2"],
        document_id="doc-1",
        document_uri="test://doc",
        doc_item_refs=["#/texts/4", "#/texts/5", "#/pictures/0"],
        page_numbers=[2, 3],
        headings=["Maintenance", "Calibration"],
    )

    [citation] = resolve_citations(["c1"], [earlier, later])

    assert citation.content == "wider window"
    assert citation.chunk_ids == ["c1", "c2"]
    assert citation.doc_item_refs == ["#/texts/4", "#/texts/5", "#/pictures/0"]
    assert citation.page_numbers == [2, 3]
    assert citation.headings == ["Maintenance", "Calibration"]
    # The window decides which figures travel with the citation.
    assert citation.picture_refs == ["#/pictures/0"]
