import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic_ai.models.function import FunctionModel

from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku_rag_app import doctor, main
from tests.support import served, settled
from tests.test_stream import _events, _input, _scripted, _session, _user, served_agent


def _configure(path: Path, databases: dict[str, str], **extra) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "lancedb": {"databases": databases},
                "databases_include": "databases.d",
                **extra,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


async def _closed(databases) -> None:
    for _ in range(200):
        if databases._client is None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the database set was not closed")


@pytest.fixture
async def stack(tmp_path, seed_database):
    """An app started from a configuration file naming alpha and beta, with
    alpha, beta and gamma seeded."""
    locations = {
        name: str(tmp_path / f"{name}.lancedb") for name in ("alpha", "beta", "gamma")
    }
    seeding = AppConfig(lancedb=LanceDBConfig(databases=locations))
    ids = {name: await seed_database(seeding, name) for name in locations}
    path = tmp_path / "haiku.rag.yaml"
    _configure(path, {"alpha": locations["alpha"], "beta": locations["beta"]})
    config = AppConfig.model_validate(load_yaml_config(path))
    async with served(config, tmp_path / "sessions.db", config_path=path) as (
        app,
        client,
    ):
        yield SimpleNamespace(
            app=app, client=client, path=path, locations=locations, ids=ids
        )


async def test_a_reload_adds_a_database_from_a_fragment(stack):
    fragments = stack.path.parent / "databases.d"
    fragments.mkdir()
    (fragments / "gamma.yaml").write_text(
        yaml.safe_dump({"lancedb": {"databases": {"gamma": stack.locations["gamma"]}}}),
        encoding="utf-8",
    )
    reloaded = await stack.client.post("/reload")
    assert reloaded.status_code == 200
    assert reloaded.json() == {
        "added": ["gamma"],
        "removed": [],
        "databases": ["alpha", "beta", "gamma"],
    }
    listed = (await stack.client.get("/api/databases")).json()["databases"]
    assert [(entry["name"], entry["available"]) for entry in listed] == [
        ("alpha", True),
        ("beta", True),
        ("gamma", True),
    ]


async def test_a_reload_removes_a_database_and_closes_the_replaced_set(stack):
    await stack.client.get("/api/databases")
    replaced = stack.app.state.configuration.current
    assert replaced._client is not None
    _configure(stack.path, {"alpha": stack.locations["alpha"]})
    reloaded = (await stack.client.post("/reload")).json()
    assert reloaded == {"added": [], "removed": ["beta"], "databases": ["alpha"]}
    await _closed(replaced)
    assert (await stack.client.get("/health")).json()["databases"] == ["alpha"]
    assert (await stack.client.get("/api/documents?source=beta")).status_code == 404


async def test_a_repointed_database_counts_as_removed_and_added(stack):
    _configure(
        stack.path,
        {"alpha": stack.locations["gamma"], "beta": stack.locations["beta"]},
    )
    reloaded = (await stack.client.post("/reload")).json()
    assert reloaded == {
        "added": ["alpha"],
        "removed": ["alpha"],
        "databases": ["alpha", "beta"],
    }
    documents = (await stack.client.get("/api/documents?source=alpha")).json()
    assert [doc["id"] for doc in documents["documents"]] == [
        stack.ids["gamma"]["document_id"]
    ]


async def test_an_unchanged_file_keeps_the_open_set(stack):
    await stack.client.get("/api/databases")
    current = stack.app.state.configuration.current
    reloaded = (await stack.client.post("/reload")).json()
    assert reloaded == {"added": [], "removed": [], "databases": ["alpha", "beta"]}
    assert stack.app.state.configuration.current is current
    assert current._client is not None


async def test_a_request_in_flight_keeps_its_set_until_it_ends(monkeypatch, stack):
    started = stack.app.state.configuration.current
    await stack.client.get("/api/databases")
    entered, release = asyncio.Event(), asyncio.Event()
    reader = started.reader

    async def held(name):
        entered.set()
        await release.wait()
        return await reader(name)

    monkeypatch.setattr(started, "reader", held)
    listing = asyncio.create_task(stack.client.get("/api/databases"))
    await entered.wait()
    _configure(stack.path, {"alpha": stack.locations["alpha"]})
    assert (await stack.client.post("/reload")).status_code == 200
    await asyncio.sleep(0.05)
    assert started._client is not None
    release.set()
    listed = (await listing).json()["databases"]
    assert [entry["available"] for entry in listed] == [True, True]
    await _closed(started)


async def test_shutdown_waits_for_a_replaced_set_to_close(monkeypatch, stack):
    replaced = stack.app.state.configuration.current
    closing, release = asyncio.Event(), asyncio.Event()
    aclose = replaced.aclose

    async def slow_close():
        closing.set()
        await release.wait()
        await aclose()

    monkeypatch.setattr(replaced, "aclose", slow_close)
    await stack.client.get("/api/databases")
    _configure(stack.path, {"alpha": stack.locations["alpha"]})
    await stack.client.post("/reload")
    await closing.wait()
    shutdown = asyncio.create_task(stack.app.state.configuration.aclose())
    await asyncio.sleep(0.05)
    assert not shutdown.done()
    release.set()
    await shutdown
    assert replaced._client is None


async def test_a_change_outside_the_databases_is_refused(stack):
    _configure(
        stack.path,
        {"alpha": stack.locations["alpha"]},
        qa={"model": {"name": "another"}},
        processing={"chunk_size": 128},
    )
    refused = await stack.client.post("/reload")
    assert refused.status_code == 409
    assert refused.json()["error"] == (
        "qa.model, processing.chunk_size changed, and only lancedb.databases is "
        "reloaded. Restart to apply the rest"
    )
    assert (await stack.client.get("/health")).json()["databases"] == [
        "alpha",
        "beta",
    ]


async def test_a_change_to_the_ingester_is_ignored(stack):
    _configure(
        stack.path,
        {"alpha": stack.locations["alpha"]},
        ingester={"workers": {"worker_count": 3}},
    )
    reloaded = await stack.client.post("/reload")
    assert reloaded.status_code == 200
    assert reloaded.json()["databases"] == ["alpha"]


@pytest.mark.parametrize(
    ("content", "detail"),
    [("lancedb: [", "while parsing"), ("lancedb:\n  uri: x\n", "databases")],
)
async def test_an_invalid_file_is_unprocessable(stack, content, detail):
    stack.path.write_text(content, encoding="utf-8")
    refused = await stack.client.post("/reload")
    assert refused.status_code == 422
    assert str(stack.path) in refused.json()["error"]
    assert detail in refused.json()["error"]
    assert (await stack.client.get("/health")).json()["databases"] == [
        "alpha",
        "beta",
    ]


async def test_an_app_without_a_configuration_file_refuses_to_reload(client):
    refused = await client.post("/reload")
    assert refused.status_code == 409
    assert "configuration file" in refused.json()["error"]


@pytest.mark.parametrize(
    "remaining", [["alpha"], ["alpha", "gamma"]], ids=["one-left", "two-left"]
)
async def test_a_turn_selecting_a_removed_database_is_refused(
    monkeypatch, stack, remaining
):
    from haiku.rag.embeddings import EmbedderWrapper

    async def embed_query(self, text):
        return [0.1] * AppConfig().embeddings.model.vector_dim

    monkeypatch.setattr(EmbedderWrapper, "embed_query", embed_query)
    client, ids = stack.client, stack.ids
    session_id = await _session(client)
    await client.patch(
        f"/api/sessions/{session_id}",
        json={
            "selection": [
                {"source": name, "id": ids[name]["document_id"]}
                for name in ("alpha", "beta")
            ]
        },
    )
    _configure(stack.path, {name: stack.locations[name] for name in remaining})
    await client.post("/reload")
    agent = await served_agent(stack.app)
    model = FunctionModel(stream_function=_scripted(ids["alpha"]["chunk_id"]))
    with agent.override(model=model):
        refused = await client.post(
            "/v1/chat/stream", json=_input(session_id, "run-1", [_user("u1")])
        )
        assert refused.status_code == 400
        assert refused.json()["error"] == (
            "the selection names 'beta', which is not configured: remove its "
            "documents from the selection"
        )
        await client.patch(
            f"/api/sessions/{session_id}",
            json={
                "selection": [{"source": "alpha", "id": ids["alpha"]["document_id"]}]
            },
        )
        answered = await client.post(
            "/v1/chat/stream", json=_input(session_id, "run-2", [_user("u2")])
        )
    summary = next(e for e in _events(answered.text) if e["type"] == "CUSTOM")
    assert summary["value"] == {"citation_status": "grounded", "violation": False}


async def test_a_repointed_database_shows_none_of_the_old_doctor_results(
    monkeypatch, stack
):
    client = stack.client
    await client.post("/api/doctor/alpha/run", params={"providers": "false"})
    assert (await settled(client, "alpha"))["status"] == "done"
    await client.post("/api/doctor/alpha/projection")
    first = await settled(client, "alpha", "/projection")
    assert first["status"] == "done"

    # The next map of the old alpha finishes after the reload.
    entered, release = threading.Event(), threading.Event()
    scan_vectors = doctor.scan_vectors

    async def held(store):
        entered.set()
        release.wait(10)
        return await scan_vectors(store)

    monkeypatch.setattr(doctor, "scan_vectors", held)
    await client.post("/api/doctor/alpha/projection")
    await asyncio.to_thread(entered.wait, 10)
    old = stack.app.state.configuration.current.scope.databases[0]
    _configure(
        stack.path,
        {"alpha": stack.locations["gamma"], "beta": stack.locations["beta"]},
    )
    assert (await client.post("/reload")).status_code == 200
    release.set()
    for _ in range(500):
        job = stack.app.state.jobs.get(old, "projection")
        if job.status == "done":
            break
        await asyncio.sleep(0.01)
    assert job.status == "done"

    assert (await client.get("/api/doctor/alpha")).json() == {"run": None}
    assert (await client.get("/api/doctor/alpha/projection")).json() == {
        "projection": None
    }
    similarity = await client.get(
        "/api/doctor/alpha/projection/similarity",
        params={
            "id": stack.ids["alpha"]["document_id"],
            "generation": job.result["generation"],
        },
    )
    assert similarity.status_code == 404
    assert (await client.get("/api/doctor/alpha/duplicates.yaml")).status_code == 404


async def test_overlapping_reloads_apply_the_newer_file(monkeypatch, stack):
    read, release = threading.Event(), threading.Event()
    load = main.load_yaml_config
    calls = 0

    def slow_first(path):
        nonlocal calls
        calls += 1
        content = load(path)
        if calls == 1:
            read.set()
            release.wait(10)
        return content

    monkeypatch.setattr(main, "load_yaml_config", slow_first)
    _configure(stack.path, {"alpha": stack.locations["alpha"]})
    first = asyncio.create_task(stack.client.post("/reload"))
    await asyncio.to_thread(read.wait, 10)
    _configure(stack.path, {"beta": stack.locations["beta"]})
    second = asyncio.create_task(stack.client.post("/reload"))
    await asyncio.sleep(0.05)
    release.set()
    await first
    await second
    assert (await stack.client.get("/health")).json()["databases"] == ["beta"]


async def test_a_replaced_set_no_request_uses_closes_at_once(stack):
    await stack.client.get("/api/databases")
    replaced = stack.app.state.configuration.current
    _configure(stack.path, {"alpha": stack.locations["alpha"]})
    await stack.app.state.configuration.reload()
    await _closed(replaced)


async def test_a_reordered_set_is_served_in_its_new_order(stack):
    _configure(
        stack.path,
        {"beta": stack.locations["beta"], "alpha": stack.locations["alpha"]},
    )
    reloaded = (await stack.client.post("/reload")).json()
    assert reloaded == {"added": [], "removed": [], "databases": ["beta", "alpha"]}
    assert (await stack.client.get("/health")).json()["databases"] == [
        "beta",
        "alpha",
    ]


async def test_a_cancelled_request_still_closes_its_replaced_set(monkeypatch, stack):
    await stack.client.get("/api/databases")
    replaced = stack.app.state.configuration.current
    entered, never = asyncio.Event(), asyncio.Event()
    closing, release = asyncio.Event(), asyncio.Event()
    aclose = replaced.aclose

    async def held(name):
        entered.set()
        await never.wait()

    async def slow_close():
        closing.set()
        await release.wait()
        await aclose()

    monkeypatch.setattr(replaced, "reader", held)
    monkeypatch.setattr(replaced, "aclose", slow_close)
    listing = asyncio.create_task(stack.client.get("/api/databases"))
    await entered.wait()
    _configure(stack.path, {"alpha": stack.locations["alpha"]})
    await stack.app.state.configuration.reload()
    listing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await listing
    await closing.wait()
    shutdown = asyncio.create_task(stack.app.state.configuration.aclose())
    await asyncio.sleep(0.05)
    assert not shutdown.done()
    release.set()
    await shutdown
    assert replaced._client is None
