from contextlib import nullcontext

import pytest

from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku_rag_app import main
from tests.support import served


async def test_health_names_the_databases_and_the_model(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert body["databases"] == ["alpha", "beta", "missing"]
    assert body["vision"] is AppConfig().qa.model.vision


async def test_a_read_failing_in_one_database_is_reported_alone(monkeypatch, client):
    import haiku.rag.store.info

    stats = haiku.rag.store.info.get_database_stats
    first = True

    async def failing_once(db, tag=None):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("stats read failed")
        return await stats(db, tag=tag)

    monkeypatch.setattr(haiku.rag.store.info, "get_database_stats", failing_once)
    entries = (await client.get("/api/databases")).json()["databases"]
    assert entries[0] == {
        "name": "alpha",
        "available": False,
        "error": "stats read failed",
    }
    assert entries[1]["available"] is True


async def test_databases_report_availability_and_size(client):
    body = (await client.get("/api/databases")).json()
    by_name = {entry["name"]: entry for entry in body["databases"]}
    assert [entry["name"] for entry in body["databases"]] == [
        "alpha",
        "beta",
        "missing",
    ]
    assert by_name["alpha"]["available"] is True
    assert by_name["alpha"]["documents"] == 1
    assert by_name["alpha"]["chunks"] == 1
    assert by_name["alpha"]["has_vector_index"] is False
    assert by_name["alpha"]["supports_image_search"] is False
    assert by_name["alpha"]["tag"] is None
    assert by_name["missing"]["available"] is False
    assert "missing" in by_name["missing"]["error"]
    assert body["vision"] is AppConfig().qa.model.vision


async def test_an_unavailable_database_opens_once_it_exists(
    seeded, seed_database, client
):
    config, _ = seeded
    first = (await client.get("/api/databases")).json()["databases"]
    assert first[2]["available"] is False
    await seed_database(config, "missing")
    second = (await client.get("/api/databases")).json()["databases"]
    assert second[2]["available"] is True


async def test_a_set_needs_a_database_named(client):
    response = await client.get(
        "/api/pictures", params={"document_id": "x", "ref": "#/pictures/0"}
    )
    assert response.status_code == 400
    assert response.json() == {"error": "name a database, one of alpha, beta, missing"}


async def test_an_unknown_database_is_not_found(client):
    response = await client.get("/api/documents", params={"source": "gamma"})
    assert response.status_code == 404
    assert response.json() == {"error": "unknown database 'gamma'"}


async def test_one_database_is_read_without_a_name(tmp_path, seed_database):
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"only": str(tmp_path / "only.lancedb")})
    )
    ids = await seed_database(config, "only")
    async with served(config, tmp_path / "sessions.db") as (_, http):
        response = await http.get(
            "/api/pictures",
            params={"document_id": ids["document_id"], "ref": "#/pictures/0"},
        )
    assert response.status_code == 200


@pytest.mark.parametrize("error", [None, RuntimeError("shutdown")])
async def test_the_lifespan_closes_the_databases(app, client, error):
    await client.get("/api/databases")
    assert app.state.databases._client is not None
    with pytest.raises(RuntimeError) if error else nullcontext():
        async with app.router.lifespan_context(app):
            if error:
                raise error
    assert app.state.databases._client is None


async def test_the_served_app_is_configured_from_the_environment(monkeypatch, tmp_path):
    import haiku.rag.telemetry

    configured: list[str] = []
    monkeypatch.setattr(
        haiku.rag.telemetry,
        "configure",
        lambda service_name: configured.append(service_name),
    )
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"only": str(tmp_path / "only.lancedb")})
    )
    monkeypatch.setattr(main, "get_config", lambda: config)
    monkeypatch.setenv("HAIKU_RAG_APP_SESSIONS", str(tmp_path / "sessions.db"))
    served = main.app()
    assert configured == ["haiku-rag-app"]
    assert served.state.config is config
    assert served.state.sessions.path == tmp_path / "sessions.db"
