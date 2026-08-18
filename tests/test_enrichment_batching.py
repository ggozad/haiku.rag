import lancedb
import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.search import _populate_image_data
from haiku.rag.store.models import DocumentItem, SearchResult


def _picture_result(document_id: str, ref: str) -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_id}-{ref}",
        document_id=document_id,
        content="body",
        score=0.9,
        doc_item_refs=[ref],
    )


async def _seed(rag: HaikuRAG, document_ids: list[str]) -> None:
    """Each document gets the same self_refs, which is what real documents do:
    `#/pictures/0` exists in every one of them."""
    for document_id in document_ids:
        await rag.document_item_repository.create_items(
            document_id,
            [
                DocumentItem(
                    document_id=document_id,
                    position=0,
                    self_ref="#/pictures/0",
                    label="picture",
                    text=f"caption for {document_id}",
                    picture_data=f"bytes-{document_id}".encode(),
                ),
            ],
        )


@pytest.fixture
def item_queries(monkeypatch):
    tally = {"n": 0}
    query = lancedb.AsyncTable.query

    def counted(self):
        if self.name == "document_items":
            tally["n"] += 1
        return query(self)

    monkeypatch.setattr(lancedb.AsyncTable, "query", counted)
    return tally


@pytest.mark.asyncio
async def test_enrichment_query_count_does_not_grow_with_documents(
    temp_db_path, item_queries
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag, [f"doc-{i}" for i in range(6)])

        results = [_picture_result("doc-0", "#/pictures/0")]
        item_queries["n"] = 0
        await _populate_image_data(rag, results)
        one_document = item_queries["n"]

        results = [_picture_result(f"doc-{i}", "#/pictures/0") for i in range(6)]
        item_queries["n"] = 0
        await _populate_image_data(rag, results)
        six_documents = item_queries["n"]

    assert (one_document, six_documents) == (2, 2), (
        f"one document took {one_document} queries, six took {six_documents}"
    )


@pytest.mark.asyncio
async def test_each_document_gets_its_own_pictures(temp_db_path):
    """self_refs collide across documents, so a batched fetch keyed on self_ref
    alone would hand one document another's picture."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag, ["doc-a", "doc-b"])

        results = [
            _picture_result("doc-a", "#/pictures/0"),
            _picture_result("doc-b", "#/pictures/0"),
        ]
        await _populate_image_data(rag, results)

    import base64

    for result, document_id in zip(results, ["doc-a", "doc-b"]):
        assert result.image_data is not None
        blob = base64.b64decode(result.image_data["#/pictures/0"])
        assert blob == f"bytes-{document_id}".encode()


@pytest.mark.asyncio
async def test_caption_ranked_results_take_at_most_four_queries(
    temp_db_path, item_queries
):
    """The worst case: results ranked on a caption, so the dependent
    caption-to-picture mapping runs too. Two for that, one for the blobs and
    their text. Still flat in document count."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        for i in range(4):
            document_id = f"doc-{i}"
            await rag.document_item_repository.create_items(
                document_id,
                [
                    DocumentItem(
                        document_id=document_id,
                        position=0,
                        self_ref="#/pictures/0",
                        label="picture",
                        text=f"caption for {document_id}",
                        picture_data=f"bytes-{document_id}".encode(),
                    ),
                    DocumentItem(
                        document_id=document_id,
                        position=1,
                        self_ref="#/texts/1",
                        label="caption",
                        text=f"figure 1 of {document_id}",
                    ),
                ],
            )

        counts = []
        for n in (1, 4):
            results = [_picture_result(f"doc-{i}", "#/texts/1") for i in range(n)]
            item_queries["n"] = 0
            await _populate_image_data(rag, results)
            counts.append(item_queries["n"])
            assert all(r.image_data for r in results)

    assert counts == [3, 3], counts


async def _seed_expandable(rag: HaikuRAG, document_ids: list[str]) -> None:
    """A section header and two text items, so expansion has something to widen
    into. Positions and self_refs repeat across documents."""
    for document_id in document_ids:
        await rag.document_item_repository.create_items(
            document_id,
            [
                DocumentItem(
                    document_id=document_id,
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text=f"Section of {document_id}",
                ),
                DocumentItem(
                    document_id=document_id,
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text=f"anchor body of {document_id}",
                ),
                DocumentItem(
                    document_id=document_id,
                    position=2,
                    self_ref="#/texts/2",
                    label="text",
                    text=f"neighbouring body of {document_id}",
                ),
            ],
        )


def _text_result(document_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_id}-anchor",
        document_id=document_id,
        content=f"anchor body of {document_id}",
        score=0.9,
        doc_item_refs=["#/texts/1"],
    )


@pytest.mark.asyncio
async def test_expansion_query_count_is_flat_in_document_count(
    temp_db_path, item_queries
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed_expandable(rag, [f"doc-{i}" for i in range(5)])

        counts = []
        for n in (1, 5):
            results = [_text_result(f"doc-{i}") for i in range(n)]
            item_queries["n"] = 0
            expanded = await rag.expand_context(results)
            counts.append(item_queries["n"])
            assert len(expanded) == n

    assert counts == [2, 2], counts


@pytest.mark.asyncio
async def test_expansion_widens_each_document_with_its_own_items(temp_db_path):
    """Positions repeat across documents, so a batched window fetch keyed on
    position alone would splice one document's text into another's context."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed_expandable(rag, ["doc-a", "doc-b"])

        expanded = await rag.expand_context(
            [_text_result("doc-a"), _text_result("doc-b")]
        )

    by_doc = {r.document_id: r.content for r in expanded}
    assert "neighbouring body of doc-a" in by_doc["doc-a"]
    assert "doc-b" not in by_doc["doc-a"]
    assert "neighbouring body of doc-b" in by_doc["doc-b"]
    assert "doc-a" not in by_doc["doc-b"]
