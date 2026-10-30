import pytest
from docling_core.types.doc.document import DoclingDocument

from haiku.rag.client import HaikuRAG


@pytest.fixture
async def manual(seeded):
    config, _ = seeded
    async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
        document = await rag.import_document(
            DoclingDocument(name="manual"),
            [],
            uri="test://alpha/manual",
            title="Field manual",
        )
    return document.id


async def test_documents_are_paged_and_counted_per_database(client, manual):
    body = (await client.get("/api/documents", params={"limit": 1})).json()
    assert sorted(doc["source"] for doc in body["documents"]) == ["alpha", "beta"]
    assert body["totals"] == {"alpha": 2, "beta": 1}
    rest = (
        await client.get(
            "/api/documents", params={"source": "alpha", "limit": 1, "offset": 1}
        )
    ).json()
    assert [doc["source"] for doc in rest["documents"]] == ["alpha"]
    assert rest["errors"] == {}
    shown = {doc["id"] for doc in body["documents"] + rest["documents"]}
    assert manual in shown and len(shown) == 3


async def test_documents_are_found_by_title_or_uri(client, manual):
    body = (await client.get("/api/documents", params={"q": "MANUAL"})).json()
    assert [doc["id"] for doc in body["documents"]] == [manual]
    assert body["totals"] == {"alpha": 1, "beta": 0}


async def test_a_page_larger_than_the_limit_is_refused(client):
    response = await client.get("/api/documents", params={"limit": 501})
    assert response.status_code == 400


async def test_documents_of_every_database_with_failures_named(client):
    body = (await client.get("/api/documents")).json()
    assert sorted((doc["source"], doc["title"]) for doc in body["documents"]) == [
        ("alpha", "alpha doc"),
        ("beta", "beta doc"),
    ]
    alpha = next(doc for doc in body["documents"] if doc["source"] == "alpha")
    assert alpha["uri"] == "test://alpha/doc"
    assert alpha["metadata"]["team"] == "alpha"
    assert list(body["errors"]) == ["missing"]


async def test_a_read_failing_in_one_database_is_reported_alone(monkeypatch, client):
    from haiku.rag.client import HaikuRAG

    listing = HaikuRAG.list_documents

    async def failing_in_alpha(self, *args, **kwargs):
        if self.source == "alpha":
            raise RuntimeError("table read failed")
        return await listing(self, *args, **kwargs)

    monkeypatch.setattr(HaikuRAG, "list_documents", failing_in_alpha)
    body = (await client.get("/api/documents")).json()
    assert [doc["source"] for doc in body["documents"]] == ["beta"]
    assert body["errors"]["alpha"] == "table read failed"
    response = await client.get("/api/documents", params={"source": "alpha"})
    assert response.status_code == 503
    assert response.json() == {"error": "table read failed"}


async def test_documents_of_an_unavailable_database_fail(client):
    response = await client.get("/api/documents", params={"source": "missing"})
    assert response.status_code == 503
    assert response.json()["error"].startswith("database 'missing'")
