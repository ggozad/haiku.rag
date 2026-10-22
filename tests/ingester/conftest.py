from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import aiosqlite
import pytest

from haiku.rag.config import QueueConfig
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo


@pytest.fixture
async def engine(tmp_path):
    eng = await open_queue(QueueConfig(path=tmp_path / "queue.db"))
    yield eng
    await eng.dispose()


@pytest.fixture
async def conn(tmp_path, engine):
    """Raw connection to the queue file for test-only SQL inspection and
    backdating. Shares the WAL database the engine writes through."""
    connection = await aiosqlite.connect(str(tmp_path / "queue.db"))
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA busy_timeout=30000")
    yield connection
    await connection.close()


@pytest.fixture
def jobs(engine):
    return JobRepo(engine)


@pytest.fixture
def sync(engine):
    return SyncStateRepo(engine)


@pytest.fixture
def use_client(monkeypatch):
    """Make IngesterApp's internally-created HaikuRAG resolve to the given
    mock client. Call again to swap the client for a later run in the same
    test."""

    def _install(client):
        """`client` serves every database, or a dict maps a database name to
        its client."""

        @asynccontextmanager
        async def _cm(scope):
            yield client[scope.names[0]] if isinstance(client, dict) else client

        # The app opens each database through `_covering`: the double stands
        # in for it.
        stub = MagicMock()
        stub._covering = lambda scope, *a, **k: _cm(scope)
        monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)

    return _install
