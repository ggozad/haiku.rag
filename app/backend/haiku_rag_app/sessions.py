import asyncio
import json
import os
import uuid
import weakref
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from haiku.rag.utils.paths import get_default_data_dir

SCHEMA_VERSION = 1
PAYLOAD_VERSION = 1
DEFAULT_TITLE = "New Session"


def default_path() -> Path:
    configured = os.environ.get("HAIKU_RAG_APP_SESSIONS")
    if configured:
        return Path(configured)
    return get_default_data_dir() / "app" / "sessions.db"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _empty() -> dict[str, Any]:
    return {
        "version": PAYLOAD_VERSION,
        "messages": [],
        "state": {},
        "selection": [],
        "runs": {},
        "message_runs": {},
        "inserted": [],
    }


def text_of(content: Any) -> str:
    """The text of an AG-UI message's content: a string, or its text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class Sessions:
    """Chat sessions in one SQLite file, one row per session.

    A session's `data` holds its AG-UI messages, the whole state map, the
    document selection, the evidence of each committed run keyed by run id, the
    run each message arrived in, and the ids of user-role messages a run
    inserted itself. The app runs as one worker, so the in-flight table here is
    the only record of which session a run holds.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None
        self._opening = asyncio.Lock()
        # Who holds each session: a run id, or a mutation's own token.
        self._in_flight: dict[str, str] = {}
        # Runs whose commit has started and not landed; their hold outlives the
        # response that started them.
        self._committing: set[tuple[str, str]] = set()

    async def _connection(self) -> aiosqlite.Connection:
        async with self._opening:
            if self._db is None:
                await self._open()
        assert self._db is not None
        return self._db

    async def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(self.path)
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("PRAGMA user_version")).fetchone()
        assert row is not None
        version = row[0]
        if version == 0:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "data TEXT NOT NULL)"
            )
            await db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            await db.commit()
        elif version != SCHEMA_VERSION:
            await db.close()
            raise HTTPException(
                503,
                f"the session store {self.path} is at schema version "
                f"{version}, and this app reads version {SCHEMA_VERSION}",
            )
        self._db = db

    async def aclose(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def in_flight(self, session_id: str) -> str | None:
        return self._in_flight.get(session_id)

    def admit(self, session_id: str, holder: str) -> None:
        """Hold the session for one run or one change, refusing any other.

        Synchronous, so no other request can pass the check between it and the
        hold.
        """
        if session_id in self._in_flight:
            raise HTTPException(409, "a run is in flight on this session")
        self._in_flight[session_id] = holder

    def tie(self, session_id: str, run_id: str, guard: object) -> None:
        """Also release a run's hold when `guard` is collected, which covers a
        response whose body never streamed. A run whose commit has started keeps
        its hold until the commit lands."""
        weakref.finalize(guard, self._abandon, session_id, run_id)

    def _abandon(self, session_id: str, run_id: str) -> None:
        if (session_id, run_id) not in self._committing:
            self.release(session_id, run_id)

    def release(self, session_id: str, holder: str) -> None:
        if self._in_flight.get(session_id) == holder:
            del self._in_flight[session_id]

    def start_commit(self, session_id: str, run_id: str, write: Awaitable[None]):
        """Run a turn's commit as its own task, holding the session until it
        lands whatever happens to the request that started it."""
        self._committing.add((session_id, run_id))
        task = asyncio.ensure_future(write)

        def landed(_: asyncio.Future) -> None:
            self._committing.discard((session_id, run_id))
            self.release(session_id, run_id)

        task.add_done_callback(landed)
        return task

    async def summaries(self) -> list[dict[str, Any]]:
        db = await self._connection()
        rows = await (
            await db.execute(
                "SELECT id, title, created_at, updated_at FROM sessions "
                "ORDER BY updated_at DESC"
            )
        ).fetchall()
        return [dict(row) for row in rows]

    async def create(self) -> dict[str, Any]:
        db = await self._connection()
        now = _now()
        session = {
            "id": str(uuid.uuid4()),
            "title": DEFAULT_TITLE,
            "created_at": now,
            "updated_at": now,
        }
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (*session.values(), json.dumps(_empty())),
        )
        await db.commit()
        return session

    async def get(self, session_id: str) -> dict[str, Any]:
        db = await self._connection()
        row = await (
            await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"no session {session_id!r}")
        data = json.loads(row["data"])
        if data.get("version") != PAYLOAD_VERSION:
            raise HTTPException(
                503,
                f"session {session_id!r} is stored at version "
                f"{data.get('version')}, and this app reads version "
                f"{PAYLOAD_VERSION}",
            )
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            **data,
        }

    async def _write(self, session: dict[str, Any]) -> None:
        db = await self._connection()
        data = {key: session[key] for key in _empty()}
        await db.execute(
            "UPDATE sessions SET title = ?, updated_at = ?, data = ? WHERE id = ?",
            (session["title"], _now(), json.dumps(data), session["id"]),
        )
        await db.commit()

    async def update(
        self, session_id: str, title: str | None, selection: list | None
    ) -> dict[str, Any]:
        token = f"change-{uuid.uuid4()}"
        self.admit(session_id, token)
        try:
            session = await self.get(session_id)
            if title is not None:
                session["title"] = title
            if selection is not None:
                session["selection"] = selection
            await self._write(session)
            return await self.get(session_id)
        finally:
            self.release(session_id, token)

    async def delete(self, session_id: str) -> None:
        token = f"change-{uuid.uuid4()}"
        self.admit(session_id, token)
        try:
            await self.get(session_id)
            db = await self._connection()
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()
        finally:
            self.release(session_id, token)

    async def commit_turn(
        self,
        session_id: str,
        run_id: str,
        user_message: dict[str, Any],
        new_messages: list[dict[str, Any]],
        state: dict[str, Any],
        evidence: dict[str, Any],
    ) -> None:
        """Store one finished turn: the user's message as received, the run's
        messages, the state the run advanced and the run's evidence."""
        session = await self.get(session_id)
        for message in [user_message, *new_messages]:
            session["message_runs"][message["id"]] = run_id
        session["inserted"].extend(
            message["id"] for message in new_messages if message["role"] == "user"
        )
        session["messages"].extend([user_message, *new_messages])
        session["state"] = state
        session["runs"][run_id] = evidence
        if session["title"] == DEFAULT_TITLE:
            text = text_of(user_message.get("content")).strip()
            if text:
                session["title"] = text if len(text) <= 60 else f"{text[:57]}..."
        await self._write(session)


def _sessions(request: Request) -> Sessions:
    return request.app.state.sessions


async def list_sessions(request: Request) -> JSONResponse:
    return JSONResponse({"sessions": await _sessions(request).summaries()})


async def create_session(request: Request) -> JSONResponse:
    return JSONResponse(await _sessions(request).create(), status_code=201)


async def get_session(request: Request) -> JSONResponse:
    sessions = _sessions(request)
    session_id = request.path_params["session_id"]
    session = await sessions.get(session_id)
    return JSONResponse({**session, "in_flight_run_id": sessions.in_flight(session_id)})


async def update_session(request: Request) -> JSONResponse:
    body = await request.json()
    title = body.get("title")
    selection = body.get("selection")
    if title is not None and not (isinstance(title, str) and title.strip()):
        raise HTTPException(400, "title must be a non-empty string")
    if selection is not None and not (
        isinstance(selection, list)
        and all(
            isinstance(ref, dict)
            and isinstance(ref.get("source"), str)
            and isinstance(ref.get("id"), str)
            for ref in selection
        )
    ):
        raise HTTPException(400, "selection must be a list of {source, id}")
    session = await _sessions(request).update(
        request.path_params["session_id"], title, selection
    )
    return JSONResponse(session)


async def delete_session(request: Request) -> Response:
    await _sessions(request).delete(request.path_params["session_id"])
    return Response(status_code=204)


async def export_session(request: Request) -> PlainTextResponse:
    session = await _sessions(request).get(request.path_params["session_id"])
    inserted = set(session["inserted"])
    lines = [f"# {session['title']}", ""]
    for message in session["messages"]:
        text = text_of(message.get("content"))
        if not text or message["id"] in inserted:
            continue
        if message["role"] == "user":
            lines += [f"**User:** {text}", ""]
        elif message["role"] == "assistant":
            lines += [f"**Assistant:** {text}", ""]
    return PlainTextResponse("\n".join(lines), media_type="text/markdown")


routes = [
    Route("/api/sessions", list_sessions, methods=["GET"]),
    Route("/api/sessions", create_session, methods=["POST"]),
    Route("/api/sessions/{session_id}", get_session, methods=["GET"]),
    Route("/api/sessions/{session_id}", update_session, methods=["PATCH"]),
    Route("/api/sessions/{session_id}", delete_session, methods=["DELETE"]),
    Route("/api/sessions/{session_id}/export", export_session, methods=["GET"]),
]
