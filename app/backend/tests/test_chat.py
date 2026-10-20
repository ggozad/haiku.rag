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


async def test_documents_of_one_database(client):
    body = (await client.get("/api/documents", params={"source": "beta"})).json()
    assert [doc["source"] for doc in body["documents"]] == ["beta"]
    assert body["errors"] == {}


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
