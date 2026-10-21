import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk


@pytest.fixture
async def structured(seeded):
    """A document with two sections: the first paragraph is held by both chunks,
    the closing note by neither."""
    config, _ = seeded
    dim = config.embeddings.model.vector_dim
    doc = DoclingDocument(name="manual")
    doc.add_heading("Bleeding", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Apply pressure.")
    doc.add_heading("Tourniquets", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Two inches above the wound.")
    doc.add_text(label=DocItemLabel.TEXT, text="Closing note.")
    async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
        document = await rag.import_document(
            doc,
            [
                Chunk(
                    content="Apply pressure.",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={"doc_item_refs": ["#/texts/0", "#/texts/1"]},
                ),
                Chunk(
                    content="Two inches above the wound.",
                    embedding=[0.1] * dim,
                    order=1,
                    metadata={"doc_item_refs": ["#/texts/1", "#/texts/3"]},
                ),
            ],
            uri="test://alpha/manual",
            title="Field manual",
        )
        chunks = await rag.chunk_repository.get_by_document_id(document.id)
    return document.id, [chunk.id for chunk in chunks]


async def test_documents_are_listed_paged_and_counted(client, structured):
    body = (
        await client.get("/api/inspect/alpha/documents", params={"limit": 1})
    ).json()
    assert body["total"] == 2
    assert len(body["documents"]) == 1
    second = (
        await client.get(
            "/api/inspect/alpha/documents", params={"limit": 1, "offset": 1}
        )
    ).json()
    assert {body["documents"][0]["id"], second["documents"][0]["id"]} >= {structured[0]}


async def test_the_text_box_and_the_filter_narrow_together(client, structured):
    found = (
        await client.get("/api/inspect/alpha/documents", params={"q": "FIELD"})
    ).json()
    assert [doc["title"] for doc in found["documents"]] == ["Field manual"]
    both = (
        await client.get(
            "/api/inspect/alpha/documents",
            params={"q": "field", "filter": "uri = 'test://alpha/doc'"},
        )
    ).json()
    assert both == {"total": 0, "documents": []}


async def test_a_bad_filter_says_why(client, structured):
    response = await client.get(
        "/api/inspect/alpha/documents", params={"filter": "nonsense ==="}
    )
    assert response.status_code == 400
    assert response.json()["error"].startswith("the filter failed:")


@pytest.mark.parametrize("params", [{"limit": "x"}, {"limit": 501}, {"offset": -1}])
async def test_paging_is_bounded(client, params):
    response = await client.get("/api/inspect/alpha/documents", params=params)
    assert response.status_code == 400


async def test_a_document_is_read_with_its_metadata(client, seeded):
    _, ids = seeded
    body = (
        await client.get(f"/api/inspect/alpha/documents/{ids['alpha']['document_id']}")
    ).json()
    assert body["title"] == "alpha doc"
    assert body["metadata"]["team"] == "alpha"
    missing = await client.get("/api/inspect/alpha/documents/nope")
    assert missing.status_code == 404


async def test_items_carry_the_chunks_that_hold_them(client, structured):
    document_id, (first, second) = structured
    items = (
        await client.get(f"/api/inspect/alpha/documents/{document_id}/items")
    ).json()["items"]
    by_text = {item["text"]: item for item in items}
    assert by_text["Bleeding"]["label"] == "section_header"
    assert by_text["Bleeding"]["chunk_ids"] == [first]
    assert by_text["Apply pressure."]["chunk_ids"] == [first, second]
    assert by_text["Closing note."]["chunk_ids"] == []
    assert [item["text"] for item in items][:2] == ["Bleeding", "Apply pressure."]


async def test_the_outline_nests_sections_with_their_chunks(client, structured):
    document_id, (first, second) = structured
    [root] = (
        await client.get(f"/api/inspect/alpha/documents/{document_id}/outline")
    ).json()["outline"]
    assert root["title"] == "Bleeding"
    assert sorted(root["chunk_ids"]) == sorted([first, second])
    [child] = root["children"]
    assert child["title"] == "Tourniquets"
    assert child["chunk_ids"] == [second]


async def test_chunks_are_paged_in_document_order(client, structured):
    document_id, (first, second) = structured
    url = f"/api/inspect/alpha/documents/{document_id}/chunks"
    page = (await client.get(url, params={"limit": 1, "offset": 1})).json()
    assert page["total"] == 2
    assert [chunk["id"] for chunk in page["chunks"]] == [second]
    assert page["chunks"][0]["order"] == 1


async def test_a_chunk_is_read_and_a_missing_one_is_not_found(client, structured):
    _, (first, _second) = structured
    chunk = (await client.get(f"/api/inspect/alpha/chunks/{first}")).json()
    assert chunk["content"] == "Apply pressure."
    assert chunk["metadata"]["doc_item_refs"] == ["#/texts/0", "#/texts/1"]
    assert (await client.get("/api/inspect/alpha/chunks/nope")).status_code == 404
    assert (
        await client.get("/api/inspect/alpha/documents/nope/items")
    ).status_code == 404


async def test_an_unavailable_database_is_reported(client):
    response = await client.get("/api/inspect/missing/documents")
    assert response.status_code == 503


async def test_a_listing_failing_without_a_filter_is_unavailable(client, monkeypatch):
    async def failing(self, *args, **kwargs):
        raise RuntimeError("table read failed")

    monkeypatch.setattr(HaikuRAG, "list_documents", failing)
    response = await client.get("/api/inspect/alpha/documents")
    assert response.status_code == 503
    assert response.json() == {"error": "table read failed"}
