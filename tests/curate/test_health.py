import sqlalchemy as sa

from haiku.rag.client import HaikuRAG
from haiku.rag.curate.store.db import health
from haiku.rag.curate.store.migrations import open_store
from haiku.rag.curate.sweep import sweep
from tests.curate.test_sweep import (
    _docling,
    _import,
    _writer_config,
    curate,  # noqa: F401
)


async def test_a_changed_database_gets_its_health_checked(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
        await rag.import_document(_docling("Text without chunks."), [], uri="u:empty")
    await sweep(config, repository)

    health = await repository.health("wiki")

    assert health is not None
    names = {check["name"] for check in health.results}
    assert {"settings_row", "vector_index", "orphaned_chunks"} <= names
    assert not names & {"duplicate_documents", "embedding_drift"}
    [unchunked] = [c for c in health.results if c["name"] == "documents_text_no_chunks"]
    assert unchunked["severity"] == "warn"
    [wiki, _] = await repository.databases(["wiki", "papers"])
    assert wiki.warned_checks >= 1 and wiki.failed_checks == 0


async def test_an_unchanged_database_keeps_its_last_health(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)
    first = await repository.health("wiki")

    await sweep(config, repository)

    assert first is not None
    assert await repository.health("wiki") == first
    assert await repository.health("nope") is None


async def test_an_unchanged_database_without_health_gets_checked(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)
    engine = await open_store(config.curate.store)
    async with engine.begin() as conn:
        await conn.execute(sa.delete(health))
    await engine.dispose()

    results = {r.database: r for r in await sweep(config, repository)}

    assert results["wiki"].status == "unchanged"
    assert await repository.health("wiki") is not None
