import json
from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


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


async def _picture(client, seeded, ref, **params):
    _, ids = seeded
    return await client.get(
        "/api/pictures",
        params={
            "source": "alpha",
            "document_id": ids["alpha"]["document_id"],
            "ref": ref,
            **params,
        },
    )


async def test_a_picture_is_served_with_its_media_type(client, seeded):
    png = await _picture(client, seeded, "#/pictures/0")
    assert png.headers["content-type"] == "image/png"
    assert png.content.startswith(b"\x89PNG")
    jpeg = await _picture(client, seeded, "#/pictures/1")
    assert jpeg.headers["content-type"] == "image/jpeg"


async def test_a_picture_in_no_known_format_is_refused(client, seeded):
    for ref in ("#/pictures/2", "#/pictures/3"):
        response = await _picture(client, seeded, ref)
        assert response.status_code == 415


async def test_a_missing_picture_is_not_found(client, seeded):
    response = await _picture(client, seeded, "#/pictures/9")
    assert response.status_code == 404


async def test_a_picture_needs_its_document_and_ref(client):
    response = await client.get("/api/pictures", params={"source": "alpha"})
    assert response.status_code == 400


async def test_a_chunk_is_drawn_on_its_page(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(f"/api/visualize/{chunk_id}", params={"source": "beta"})
    ).json()
    assert len(body["images"]) == 1
    assert body["message"] is None


async def test_a_chunk_is_drawn_from_the_refs_the_model_saw(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(
            f"/api/visualize/{chunk_id}",
            params={"source": "beta", "refs": json.dumps(["#/texts/0"])},
        )
    ).json()
    assert len(body["images"]) == 1


async def test_refs_that_draw_nothing_say_so(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(
            f"/api/visualize/{chunk_id}",
            params={"source": "beta", "refs": json.dumps(["#/texts/99"])},
        )
    ).json()
    assert body["images"] == []
    assert body["message"] == "No visual grounding available"


async def test_refs_must_be_a_json_list(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    for refs in ("not json", json.dumps({"a": 1})):
        response = await client.get(
            f"/api/visualize/{chunk_id}", params={"source": "beta", "refs": refs}
        )
        assert response.status_code == 400


async def test_a_chunk_of_another_database_is_not_found(client, seeded):
    _, ids = seeded
    response = await client.get(
        f"/api/visualize/{ids['beta']['chunk_id']}", params={"source": "alpha"}
    )
    assert response.status_code == 404


async def _answer(
    _messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str]:
    yield "An answer."


def _events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def test_the_stream_snapshots_the_state_before_the_run_ends(app, client):
    run_input = {
        "threadId": "thread",
        "runId": "run",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "Hello?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    with app.state.agent.override(model=FunctionModel(stream_function=_answer)):
        response = await client.post("/v1/chat/stream", json=run_input)
    types = [event["type"] for event in _events(response.text)]
    assert types[-2:] == ["STATE_SNAPSHOT", "RUN_FINISHED"]
    [snapshot] = [e for e in _events(response.text) if e["type"] == "STATE_SNAPSHOT"]
    assert "rag" in snapshot["snapshot"]
    assert "TEXT_MESSAGE_CONTENT" in types


async def test_events_reach_the_client_while_the_model_is_still_answering(app):
    import asyncio

    from pydantic_ai.ui.ag_ui import AGUIAdapter

    from haiku_rag_app.chat import ChatDeps, chat_events

    release = asyncio.Event()

    async def held(
        _messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str]:
        yield "First words"
        await release.wait()
        yield " and the rest."

    run_input = AGUIAdapter.build_run_input(
        json.dumps(
            {
                "threadId": "thread",
                "runId": "run",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "Hello?"}],
                "tools": [],
                "context": [],
                "forwardedProps": {},
            }
        ).encode()
    )
    adapter = AGUIAdapter(agent=app.state.agent, run_input=run_input)
    with app.state.agent.override(model=FunctionModel(stream_function=held)):
        stream = adapter.encode_stream(chat_events(adapter, ChatDeps(state={})))
        seen: list[str] = []
        async for chunk in stream:
            seen.extend(event["type"] for event in _events(chunk))
            if "TEXT_MESSAGE_CONTENT" in seen:
                break
        assert "RUN_FINISHED" not in seen
        release.set()
        async for chunk in stream:
            seen.extend(event["type"] for event in _events(chunk))
    assert seen[-2:] == ["STATE_SNAPSHOT", "RUN_FINISHED"]
