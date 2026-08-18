import lancedb
import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.models import DocumentItem
from haiku.rag.store.repositories.document_item import DocumentItemRepository


async def _seed(repo: DocumentItemRepository, document_id: str) -> None:
    """A picture at position 0 with its caption at position 1, the docling
    layout. Every document uses the same self_refs."""
    await repo.create_items(
        document_id,
        [
            DocumentItem(
                document_id=document_id,
                position=0,
                self_ref="#/pictures/0",
                label="picture",
                text=f"caption text {document_id}",
                picture_data=f"bytes-{document_id}".encode(),
            ),
            DocumentItem(
                document_id=document_id,
                position=1,
                self_ref="#/texts/1",
                label="caption",
                text=f"figure 1 {document_id}",
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
async def test_pictures_grouped_keeps_documents_apart(temp_db_path, item_queries):
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentItemRepository(store)
        await _seed(repo, "doc-a")
        await _seed(repo, "doc-b")

        item_queries["n"] = 0
        blobs, texts = await repo.get_pictures_grouped(
            {"doc-a": ["#/pictures/0"], "doc-b": ["#/pictures/0"]}
        )

        assert item_queries["n"] == 1
        assert blobs == {
            "doc-a": {"#/pictures/0": b"bytes-doc-a"},
            "doc-b": {"#/pictures/0": b"bytes-doc-b"},
        }
        assert texts == {
            "doc-a": {"#/pictures/0": "caption text doc-a"},
            "doc-b": {"#/pictures/0": "caption text doc-b"},
        }


@pytest.mark.asyncio
async def test_pictures_grouped_fetches_only_requested_documents(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentItemRepository(store)
        await _seed(repo, "doc-a")
        await _seed(repo, "doc-b")

        blobs, _ = await repo.get_pictures_grouped({"doc-a": ["#/pictures/0"]})

        assert blobs == {"doc-a": {"#/pictures/0": b"bytes-doc-a"}}


@pytest.mark.asyncio
async def test_caption_picture_refs_grouped_uses_two_queries(
    temp_db_path, item_queries
):
    """The stages are dependent: the caption's position is what finds its
    picture, so this is two queries however many documents are asked for."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentItemRepository(store)
        for document_id in ("doc-a", "doc-b", "doc-c"):
            await _seed(repo, document_id)

        item_queries["n"] = 0
        got = await repo.get_caption_picture_refs_grouped(
            {did: ["#/texts/1"] for did in ("doc-a", "doc-b", "doc-c")}
        )

        assert item_queries["n"] == 2
        for document_id in ("doc-a", "doc-b", "doc-c"):
            assert got[document_id] == {"#/texts/1": "#/pictures/0"}


@pytest.mark.asyncio
async def test_grouped_calls_with_nothing_asked_for_do_not_query(
    temp_db_path, item_queries
):
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentItemRepository(store)

        item_queries["n"] = 0
        assert await repo.get_pictures_grouped({}) == ({}, {})
        assert await repo.get_pictures_grouped({"doc-a": []}) == ({}, {})
        assert await repo.get_caption_picture_refs_grouped({}) == {}

        assert item_queries["n"] == 0


@pytest.mark.asyncio
async def test_caption_picture_refs_grouped_ignores_non_picture_predecessors(
    temp_db_path,
):
    """A caption maps to a picture only. A table's caption, and an ordinary text
    reference, map to nothing."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentItemRepository(store)
        await repo.create_items(
            "doc-1",
            [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/tables/0",
                    label="table",
                    text="a table",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/table-caption",
                    label="caption",
                    text="Table 1",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/plain",
                    label="text",
                    text="ordinary prose",
                ),
            ],
        )

        got = await repo.get_caption_picture_refs_grouped(
            {"doc-1": ["#/texts/table-caption", "#/texts/plain"]}
        )

        assert got == {}
