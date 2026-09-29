import json
import logging
import shutil

import numpy as np
import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import (
    AppConfig,
    CurateConfig,
    CurateStoreConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    LanceDBConfig,
)
from haiku.rag.curate import sweep as sweep_module
from haiku.rag.curate.chunks import chunk_text_hash
from haiku.rag.curate.store.migrations import open_store
from haiku.rag.curate.store.models import SweepStatus
from haiku.rag.curate.store.repository import CurateRepository
from haiku.rag.curate.sweep import sweep
from haiku.rag.store.models import Chunk
from tests.conftest import capture_logs

DIM = 8


def _writer_config() -> AppConfig:
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(provider="ollama", name="test", vector_dim=DIM)
        )
    )


def _curate_config(databases: dict, store_path) -> AppConfig:
    """A config that places the databases and names an embedder that cannot be built."""
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(provider="not-installed", name="x")
        ),
        lancedb=LanceDBConfig(databases={k: str(v) for k, v in databases.items()}),
        curate=CurateConfig(store=CurateStoreConfig(path=store_path)),
    )


def _docling(text: str) -> DoclingDocument:
    doc = DoclingDocument(name="doc")
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


def _chunks(texts: list[str], axes: list[int]) -> list[Chunk]:
    eye = np.eye(DIM)
    return [
        Chunk(content=text, embedding=eye[axis].tolist(), order=n)
        for n, (text, axis) in enumerate(zip(texts, axes, strict=True))
    ]


async def _import(rag: HaikuRAG, uri: str, texts: list[str], axes: list[int]) -> str:
    doc = await rag.import_document(
        _docling(" ".join(texts)), _chunks(texts, axes), uri=uri
    )
    assert doc.id is not None
    return doc.id


async def _rewrite(rag: HaikuRAG, doc_id: str, texts: list[str], axes: list[int]):
    await rag.update_document(
        doc_id, docling_document=_docling(" ".join(texts)), chunks=_chunks(texts, axes)
    )


async def _set_metadata(rag: HaikuRAG, doc_id: str, metadata: dict) -> None:
    await rag.store.document_meta_table.update(
        where=f"id = '{doc_id}'", updates={"metadata": json.dumps(metadata)}
    )


async def _edit_settings(path, edit) -> None:
    async with HaikuRAG(path, _writer_config()) as rag:
        settings = json.loads(
            (await rag.store.settings_table.query().to_list())[0]["settings"]
        )
        edit(settings)
        await rag.store.settings_table.update(
            where="id = 'settings'", updates={"settings": json.dumps(settings)}
        )


@pytest.fixture
async def curate(tmp_path):
    """Two databases, the curate config placing them, and the curate store."""
    paths = {"wiki": tmp_path / "wiki.lancedb", "papers": tmp_path / "papers.lancedb"}
    for path in paths.values():
        async with HaikuRAG(path, _writer_config(), create=True):
            pass
    config = _curate_config(paths, tmp_path / "curate.db")
    engine = await open_store(config.curate.store)
    try:
        yield paths, config, CurateRepository(engine)
    finally:
        await engine.dispose()


def _by_database(results):
    return {result.database: result for result in results}


async def test_first_sweep_fingerprints_every_document(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(
            rag, "file:///wiki/a.pdf", ["Alpha beta gamma.", "Delta."], [0, 1]
        )
        await _set_metadata(rag, doc_id, {"md5": "m1", "department": "ops"})

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.OK
    assert results["wiki"].embedder == f"ollama:test:{DIM}"
    assert results["papers"].status is SweepStatus.OK
    [fingerprint] = await repository.history("wiki", doc_id)
    assert fingerprint.uri == "file:///wiki/a.pdf"
    assert fingerprint.md5 == "m1"
    assert fingerprint.metadata_keys == ["department", "md5"]
    assert fingerprint.chunks == 2 and fingerprint.embedded_chunks == 2
    assert fingerprint.chars == len("Alpha beta gamma.") + len("Delta.")
    assert fingerprint.replacement_chars == 0
    assert fingerprint.chunk_stats["short_share"] == 1.0
    assert fingerprint.embedder == f"ollama:test:{DIM}"
    assert fingerprint.ended_sweep is None
    assert fingerprint.centroid is not None
    centroid = np.frombuffer(fingerprint.centroid, dtype=np.float32)
    np.testing.assert_allclose(centroid[:2], [2**-0.5, 2**-0.5], rtol=1e-6)
    assert sorted(await repository.chunk_text_hashes(fingerprint.id)) == sorted(
        [chunk_text_hash("Alpha beta gamma."), chunk_text_hash("Delta.")]
    )


async def test_unchanged_database_is_skipped(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.UNCHANGED
    assert len(await repository.history("wiki", doc_id)) == 1


async def test_rewritten_content_becomes_a_new_fingerprint(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, ["Price \ufffd\ufffd list\ufffd"], [3])

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.OK
    first, second = await repository.history("wiki", doc_id)
    assert first.ended_sweep is not None and not first.deleted
    assert second.ended_sweep is None
    assert second.replacement_chars == 3
    assert await repository.chunk_text_hashes(first.id) == []
    assert await repository.chunk_text_hashes(second.id) == [
        chunk_text_hash("Price \ufffd\ufffd list\ufffd")
    ]


async def test_content_returning_to_an_earlier_revision_is_recorded(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)
    for texts, axis in ((["Beta."], 1), (["Alpha."], 0)):
        async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
            await _rewrite(rag, doc_id, texts, [axis])
        await sweep(config, repository)

    history = await repository.history("wiki", doc_id)

    assert [f.chars for f in history] == [6, 5, 6]
    assert [f.ended_sweep is None for f in history] == [False, False, True]
    assert history[0].centroid == history[2].centroid != history[1].centroid


async def test_rolled_revision_with_unchanged_bytes_refreshes_in_place(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
        await _set_metadata(rag, doc_id, {"md5": "m1", "source_revision": "r1"})
    await sweep(config, repository)
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _set_metadata(
            rag, doc_id, {"md5": "m1", "source_revision": "r2", "owner": "x"}
        )

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.OK
    [fingerprint] = await repository.history("wiki", doc_id)
    assert fingerprint.source_revision == "r2"
    assert fingerprint.metadata_keys == ["md5", "owner", "source_revision"]


async def test_deleted_and_readded_document(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await rag.delete_document(doc_id)
    await sweep(config, repository)

    [gone] = await repository.history("wiki", doc_id)
    assert gone.deleted and gone.ended_sweep is not None

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        readded = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)

    [current] = await repository.history("wiki", readded)
    assert current.ended_sweep is None and not current.deleted


async def test_embedder_change_rebaselines_every_document(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    await sweep(config, repository)
    await _edit_settings(
        paths["wiki"], lambda s: s["embeddings"]["model"].update(name="other")
    )

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].rebaseline
    assert results["wiki"].embedder == f"ollama:other:{DIM}"
    first, second = await repository.history("wiki", doc_id)
    assert first.change_key == second.change_key
    assert (first.embedder, second.embedder) == (
        f"ollama:test:{DIM}",
        f"ollama:other:{DIM}",
    )


async def test_write_during_a_sweep_discards_the_database(curate, monkeypatch):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    read = sweep_module._read_changed

    async def read_then_write(store, *args, **kwargs):
        result = await read(store, *args, **kwargs)
        if store.db_path == paths["wiki"]:
            async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
                await _rewrite(rag, doc_id, ["Beta."], [1])
        return result

    monkeypatch.setattr(sweep_module, "_read_changed", read_then_write)

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.MOVED
    assert await repository.history("wiki", doc_id) == []
    assert await repository.last_ok_sweep("wiki") is None


async def test_unavailable_database_is_an_error_and_others_proceed(curate, tmp_path):
    paths, config, repository = curate
    config.lancedb.databases["gone"] = str(tmp_path / "gone.lancedb")

    results = _by_database(await sweep(config, repository))

    assert results["gone"].status is SweepStatus.ERROR
    assert results["gone"].error is not None
    assert "gone.lancedb" not in results["gone"].error
    assert results["wiki"].status is SweepStatus.OK


async def test_sweep_covers_only_the_selected_databases(curate):
    paths, config, repository = curate
    config.curate.databases = ["papers"]

    results = _by_database(await sweep(config, repository))

    assert list(results) == ["papers"]


async def test_database_needing_migration_is_an_error(curate):
    paths, config, repository = curate
    await _edit_settings(paths["wiki"], lambda s: s.update(version="0.75.0"))

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.ERROR
    assert "migrate" in (results["wiki"].error or "")


async def test_documents_without_embedded_chunks_have_no_centroid(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        empty = (await rag.import_document(_docling("Nothing."), [], uri="u:e")).id
        zero = (
            await rag.import_document(
                _docling("Zero."),
                [Chunk(content="Zero.", embedding=[0.0] * DIM, order=0)],
                uri="u:z",
            )
        ).id
    assert empty is not None and zero is not None

    await sweep(config, repository)

    [no_chunks] = await repository.history("wiki", empty)
    assert no_chunks.chunks == 0 and no_chunks.centroid is None
    assert no_chunks.chunk_stats["p50"] is None
    [unembedded] = await repository.history("wiki", zero)
    assert unembedded.chunks == 1 and unembedded.embedded_chunks == 0
    assert unembedded.centroid is None


async def test_database_without_a_recorded_embedder(curate):
    paths, config, repository = curate
    await _edit_settings(paths["wiki"], lambda s: s.pop("embeddings"))

    results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.OK
    assert results["wiki"].embedder is None


async def test_unreadable_database_is_an_error_and_later_databases_proceed(curate):
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    shutil.rmtree(paths["wiki"] / "chunks.lance" / "data")

    with capture_logs(sweep_module.logger, logging.ERROR) as records:
        results = _by_database(await sweep(config, repository))

    assert results["wiki"].status is SweepStatus.ERROR
    assert results["wiki"].error is not None
    assert results["wiki"].error.startswith("sweep failed: ")
    assert str(paths["wiki"]) not in results["wiki"].error
    assert results["papers"].status is SweepStatus.OK
    assert [r.getMessage() for r in records] == ["Sweeping database 'wiki' failed"]
