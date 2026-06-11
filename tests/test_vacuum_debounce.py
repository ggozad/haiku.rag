import asyncio

import pytest

import haiku.rag.client as client_mod
from haiku.rag.client import HaikuRAG


@pytest.mark.asyncio
async def test_schedule_vacuum_is_debounced(temp_db_path, monkeypatch):
    """Rapid writes within the throttle window schedule only one background
    vacuum; once the interval elapses, a new one is scheduled."""
    t = {"now": 1000.0}
    monkeypatch.setattr(client_mod, "monotonic", lambda: t["now"])

    async with HaikuRAG(temp_db_path, create=True) as client:
        calls: list[int] = []

        async def fake_vacuum(*_a, **_k):
            calls.append(1)

        monkeypatch.setattr(client.store, "vacuum", fake_vacuum)

        for _ in range(3):
            client._schedule_vacuum()
        await asyncio.gather(*client._vacuum_tasks)
        assert len(calls) == 1  # debounced within the interval

        t["now"] += client_mod._VACUUM_MIN_INTERVAL_S + 1
        client._schedule_vacuum()
        await asyncio.gather(*client._vacuum_tasks)
        assert len(calls) == 2  # interval elapsed -> a new vacuum scheduled


@pytest.mark.asyncio
async def test_debounced_writes_still_collapse_on_close(temp_db_path, monkeypatch):
    """Even when scheduled vacuums after the first are debounced, the writes are
    marked dirty so the close-time drain runs a final collapse."""
    t = {"now": 1000.0}
    monkeypatch.setattr(client_mod, "monotonic", lambda: t["now"])
    calls: list[int] = []

    async with HaikuRAG(temp_db_path, create=True) as client:

        async def fake_vacuum(*_a, **_k):
            calls.append(1)

        monkeypatch.setattr(client.store, "vacuum", fake_vacuum)

        client._schedule_vacuum()  # schedules the first background pass
        client._schedule_vacuum()  # debounced (no task)

        await client._await_vacuum_tasks()
        # one scheduled background pass + one final collapse on drain
        assert len(calls) == 2
        assert client._vacuum_dirty is False
