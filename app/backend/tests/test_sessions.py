import asyncio
import json
import sqlite3

import pytest
from starlette.exceptions import HTTPException

from haiku_rag_app.sessions import Sessions, default_path, text_of


class Guard:
    """Stands for the response a hold is tied to."""


async def test_a_session_is_created_listed_read_and_deleted(client):
    created = (await client.post("/api/sessions")).json()
    assert created["title"] == "New Session"
    listed = (await client.get("/api/sessions")).json()["sessions"]
    assert [s["id"] for s in listed] == [created["id"]]
    session = (await client.get(f"/api/sessions/{created['id']}")).json()
    assert session["messages"] == []
    assert session["selection"] == []
    assert session["in_flight_run_id"] is None
    assert (await client.delete(f"/api/sessions/{created['id']}")).status_code == 204
    assert (await client.get(f"/api/sessions/{created['id']}")).status_code == 404


async def test_a_session_is_renamed_and_scoped(client):
    session_id = (await client.post("/api/sessions")).json()["id"]
    response = await client.patch(
        f"/api/sessions/{session_id}",
        json={"title": "Bleeding", "selection": [{"source": "alpha", "id": "a"}]},
    )
    assert response.json()["title"] == "Bleeding"
    assert response.json()["selection"] == [{"source": "alpha", "id": "a"}]


@pytest.mark.parametrize(
    "body",
    [
        {"title": ""},
        {"title": 3},
        {"selection": "all"},
        {"selection": [{"source": "alpha"}]},
    ],
)
async def test_a_malformed_update_is_refused(client, body):
    session_id = (await client.post("/api/sessions")).json()["id"]
    response = await client.patch(f"/api/sessions/{session_id}", json=body)
    assert response.status_code == 400


async def test_an_unknown_session_is_not_found(client):
    assert (await client.get("/api/sessions/nope")).status_code == 404
    assert (await client.delete("/api/sessions/nope")).status_code == 404
    response = await client.patch("/api/sessions/nope", json={"title": "x"})
    assert response.status_code == 404


async def test_a_session_held_by_a_run_refuses_changes(app, client):
    session_id = (await client.post("/api/sessions")).json()["id"]
    app.state.sessions.admit(session_id, "run-1")
    session = (await client.get(f"/api/sessions/{session_id}")).json()
    assert session["in_flight_run_id"] == "run-1"
    patched = await client.patch(f"/api/sessions/{session_id}", json={"title": "x"})
    assert patched.status_code == 409
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 409
    with pytest.raises(HTTPException) as refused:
        app.state.sessions.admit(session_id, "run-2")
    assert refused.value.status_code == 409
    app.state.sessions.release(session_id, "run-1")
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204


async def test_a_hold_whose_response_never_streamed_is_released(tmp_path):
    import gc

    sessions = Sessions(tmp_path / "sessions.db")
    guard = Guard()
    sessions.admit("s", "run-1")
    sessions.tie("s", "run-1", guard)
    sessions.release("s", "run-other")
    assert sessions.in_flight("s") == "run-1"
    del guard
    gc.collect()
    assert sessions.in_flight("s") is None


async def test_a_change_holds_the_session_against_a_run(app, client, monkeypatch):
    session_id = (await client.post("/api/sessions")).json()["id"]
    sessions: Sessions = app.state.sessions
    reading, release = asyncio.Event(), asyncio.Event()
    get = Sessions.get

    async def held_get(self, *args):
        reading.set()
        await release.wait()
        return await get(self, *args)

    monkeypatch.setattr(Sessions, "get", held_get)
    change = asyncio.ensure_future(sessions.update(session_id, "renamed", None))
    await reading.wait()
    with pytest.raises(HTTPException) as refused:
        sessions.admit(session_id, "run-1")
    assert refused.value.status_code == 409
    release.set()
    await change
    assert sessions.in_flight(session_id) is None


async def test_concurrent_first_requests_open_one_connection(tmp_path, monkeypatch):
    import aiosqlite

    opened = []
    connect = aiosqlite.connect

    def counting(*args, **kwargs):
        opened.append(args)
        return connect(*args, **kwargs)

    monkeypatch.setattr(aiosqlite, "connect", counting)
    sessions = Sessions(tmp_path / "sessions.db")
    await asyncio.gather(sessions.summaries(), sessions.summaries())
    assert len(opened) == 1
    await sessions.aclose()


async def test_an_export_holds_what_the_people_said(app, client):
    session_id = (await client.post("/api/sessions")).json()["id"]
    await app.state.sessions.commit_turn(
        session_id,
        "run-1",
        {"id": "u1", "role": "user", "content": "Where?"},
        [
            {"id": "a1", "role": "assistant", "content": "Here."},
            {"id": "r1", "role": "user", "content": "Cite what you used."},
            {"id": "a2", "role": "assistant", "content": ""},
        ],
        {},
        {"citations": [], "program": None, "summary": {}},
    )
    response = await client.get(f"/api/sessions/{session_id}/export")
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == "# Where?\n\n**User:** Where?\n\n**Assistant:** Here.\n"


async def test_a_store_from_another_version_is_refused(tmp_path):
    path = tmp_path / "sessions.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 7")
    with pytest.raises(HTTPException) as refused:
        await Sessions(path).summaries()
    assert refused.value.status_code == 503
    assert "schema version 7" in refused.value.detail


async def test_a_session_from_another_version_is_refused(tmp_path):
    sessions = Sessions(tmp_path / "sessions.db")
    session_id = (await sessions.create())["id"]
    with sqlite3.connect(sessions.path) as db:
        db.execute(
            "UPDATE sessions SET data = ? WHERE id = ?",
            (json.dumps({"version": 9}), session_id),
        )
    with pytest.raises(HTTPException) as refused:
        await sessions.get(session_id)
    assert "version 9" in refused.value.detail
    await sessions.aclose()


def test_the_store_lives_where_the_environment_says(monkeypatch, tmp_path):
    monkeypatch.setenv("HAIKU_RAG_APP_SESSIONS", str(tmp_path / "s.db"))
    assert default_path() == tmp_path / "s.db"
    monkeypatch.delenv("HAIKU_RAG_APP_SESSIONS")
    assert default_path().name == "sessions.db"


def test_text_is_read_from_strings_and_text_parts():
    assert text_of("plain") == "plain"
    assert text_of([{"type": "text", "text": "a"}, {"type": "image"}, "x"]) == "a"
    assert text_of(None) == ""


async def test_the_same_run_id_on_two_sessions_is_two_holds(tmp_path):
    import gc

    sessions = Sessions(tmp_path / "sessions.db")
    release = asyncio.Event()

    async def held_write():
        await release.wait()

    sessions.admit("one", "run-1")
    sessions.admit("two", "run-1")
    commit = sessions.start_commit("one", "run-1", held_write())
    guard = Guard()
    sessions.tie("two", "run-1", guard)
    del guard
    gc.collect()
    assert sessions.in_flight("two") is None
    assert sessions.in_flight("one") == "run-1"
    sessions.admit("two", "run-1")
    release.set()
    await commit
    assert sessions.in_flight("one") is None
    assert sessions.in_flight("two") == "run-1"
