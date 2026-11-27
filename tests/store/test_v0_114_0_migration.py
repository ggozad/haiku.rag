import asyncio
import hashlib
import logging
import os
import threading
from collections.abc import Sequence

import lance
import pyarrow as pa
import pytest

from haiku.rag.config.models import AppConfig
from haiku.rag.store.engine import COMPACTION_TARGET_KEY, ConnectionMode, Store
from haiku.rag.store.exceptions import MigrationRequiredError
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.schema import (
    ensure_indexes,
    get_documents_arrow_schema,
    is_blob,
)
from haiku.rag.store.upgrades import v0_114_0
from tests.conftest import capture_logs

MB = 1024**2
# Inline, packed into a shared sidecar, and in a sidecar of its own.
TIERS = [1_000, 200_000, 9 * MB]


def _previous_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("content", pa.string()),
            pa.field("docling_document", pa.large_binary()),
            pa.field("docling_version", pa.string()),
            pa.field("original", pa.large_binary()),
            pa.field("original_md5", pa.string()),
        ]
    )


async def at_previous_release(
    store: Store, originals: Sequence[bytes | None], per_write: int = 1
) -> list[str]:
    """Documents with `originals`, in a `documents` table stored as the release
    before 0.114.0 stored it: lance format 2.0, `original` as `large_binary`."""
    repo = DocumentRepository(store)
    created = await repo.create([Document(content=str(i)) for i in originals])
    ids = [d.id for d in created if d.id is not None]
    rows = [
        {
            "id": i,
            "content": f"doc {n}",
            "docling_document": b"structure",
            "docling_version": "1",
            "original": body,
            "original_md5": None if body is None else hashlib.md5(body).hexdigest(),
        }
        for n, (i, body) in enumerate(zip(ids, originals))
    ]
    uri = (await store.documents_table.to_lance()).uri
    options = store._config.lancedb.storage_options or None
    for start in range(0, len(rows), per_write):
        lance.write_dataset(
            pa.Table.from_pylist(rows[start : start + per_write], _previous_schema()),
            uri,
            mode="overwrite" if start == 0 else "append",
            data_storage_version="2.0",
            storage_options=options,
        )
    await store.documents_table.checkout_latest()
    await ensure_indexes(
        store.documents_table, "documents", store._config.search.fts_language
    )
    await store._write_settings(version="0.113.2", schema_version="0.113.0")
    return ids


def _config(target_bytes: int | None = None) -> AppConfig:
    config = AppConfig()
    config.storage.auto_vacuum = False
    if target_bytes is not None:
        config.storage.compaction_target_bytes = target_bytes
    return config


async def _migrate(path, config: AppConfig | None = None) -> None:
    async with Store(path, config=config or _config(), skip_migration_check=True) as s:
        await s.migrate()


def test_rows_are_read_with_their_originals_as_blob_values(tmp_path):
    """Called here as well: lance reads the rows on a thread it starts itself,
    which coverage does not trace."""
    rows = [
        {
            "id": i,
            "content": "c",
            "docling_document": None,
            "docling_version": None,
            "original": body,
            "original_md5": None,
        }
        for i, body in (("a", b"%PDF-a"), ("b", None))
    ]
    dataset = lance.write_dataset(
        pa.Table.from_pylist(rows, _previous_schema()), tmp_path / "d.lance"
    )
    schema = get_documents_arrow_schema()

    batches = list(v0_114_0._as_blobs(dataset, schema))

    assert all(b.schema == schema for b in batches)
    originals = [r["original"] for b in batches for r in b.to_pylist()]
    assert [o if o is None else o["data"] for o in originals] == [b"%PDF-a", None]


async def test_originals_become_blobs_at_format_2_2(temp_db_path):
    bodies = [os.urandom(n) for n in TIERS] + [None]
    async with Store(temp_db_path, config=_config(), create=True) as store:
        ids = await at_previous_release(store, bodies)
        indexes = sorted(i.name for i in await store.documents_table.list_indices())

    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path, read_only=True):
            pass
    await _migrate(temp_db_path)

    async with Store(temp_db_path, read_only=True) as store:
        table = store.documents_table
        dataset = await table.to_lance()
        assert dataset.data_storage_version == "2.2"
        assert is_blob((await table.schema()).field("original"))
        assert sorted(i.name for i in await table.list_indices()) == indexes
        repo = DocumentRepository(store)
        for i, body in zip(ids, bodies):
            assert await repo.get_original(i) == body
            document = await repo.get_docling_data(i)
            assert document is not None
            assert document.docling_document == b"structure"
        assert len(dataset.versions()) == 1


async def test_the_rewrite_keeps_fragments_within_the_compaction_target(
    temp_db_path,
):
    """Twenty 200 KB originals at a 1 MB target: about five rows a fragment,
    where one overwrite with no limit writes a single fragment."""
    config = _config(target_bytes=1 * MB)
    async with Store(temp_db_path, config=config, create=True) as store:
        await at_previous_release(store, [os.urandom(200_000) for _ in range(20)])

    await _migrate(temp_db_path, config)

    async with Store(temp_db_path, config=config, read_only=True) as store:
        fragments = (await store.documents_table.to_lance()).get_fragments()
    assert len(fragments) > 1
    assert max(f.physical_rows for f in fragments) <= 5


async def test_the_migration_keeps_the_fragments_it_writes(temp_db_path):
    """The 400 KB original sets a target of two rows. Every rewritten fragment
    mixes it with narrower ones, so sizing the new layout alone gives a higher
    target, and a vacuum with nothing stored would merge them all."""
    config = _config(target_bytes=1 * MB)
    bodies = [os.urandom(400_000)] + [os.urandom(50_000) for _ in range(19)]
    async with Store(temp_db_path, config=config, create=True) as store:
        await at_previous_release(store, bodies)

    await _migrate(temp_db_path, config)

    async with Store(temp_db_path, config=config, read_only=True) as store:
        dataset = await store.documents_table.to_lance()
    assert dataset.config()[COMPACTION_TARGET_KEY] == "2"
    assert max(f.physical_rows for f in dataset.get_fragments()) == 2


async def test_an_oversized_original_is_written_alone(temp_db_path):
    """Above half of the 1 MB target, a row would push any fragment it shares
    past the target."""
    config = _config(target_bytes=1 * MB)
    bodies = [os.urandom(600_000)] + [os.urandom(50_000) for _ in range(10)]
    async with Store(temp_db_path, config=config, create=True) as store:
        await at_previous_release(store, bodies)

    await _migrate(temp_db_path, config)

    async with Store(temp_db_path, config=config, read_only=True) as store:
        dataset = await store.documents_table.to_lance()
    holding = [
        f.physical_rows
        for f in dataset.get_fragments()
        if any(
            d is not None and d["size"] > MB // 2
            for d in f.to_table(columns=["original"])["original"].to_pylist()
        )
    ]
    assert holding == [1]


@pytest.mark.parametrize("failing", ["update_config", "write_dataset"])
async def test_a_failed_rewrite_leaves_the_table_as_it_was(
    temp_db_path, monkeypatch, failing
):
    body = os.urandom(1_000)
    async with Store(temp_db_path, config=_config(), create=True) as store:
        await at_previous_release(store, [body])

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    if failing == "write_dataset":
        monkeypatch.setattr(v0_114_0.lance, "write_dataset", fail)
    else:
        monkeypatch.setattr(lance.LanceDataset, "update_config", fail)
    with pytest.raises(OSError, match="disk full"):
        await _migrate(temp_db_path)
    monkeypatch.undo()

    async with Store(temp_db_path, skip_migration_check=True) as store:
        rows = await store.documents_table.query().select(["original"]).to_list()
        original = (await store.documents_table.schema()).field("original")
    assert original.type == pa.large_binary()
    assert [r["original"] for r in rows] == [body]


async def test_a_cancelled_migration_finishes_the_rewrite_first(
    temp_db_path, monkeypatch
):
    body = os.urandom(200_000)
    async with Store(temp_db_path, config=_config(), create=True) as store:
        [doc_id] = await at_previous_release(store, [body])

    writing, release = threading.Event(), threading.Event()
    write_dataset = v0_114_0.lance.write_dataset

    def slow_write(*args, **kwargs):
        writing.set()
        release.wait(10)
        return write_dataset(*args, **kwargs)

    monkeypatch.setattr(v0_114_0.lance, "write_dataset", slow_write)
    async with Store(
        temp_db_path, config=_config(), skip_migration_check=True
    ) as store:
        migration = asyncio.create_task(store.migrate())
        await asyncio.to_thread(writing.wait, 10)
        migration.cancel()
        await asyncio.sleep(0.1)
        assert not migration.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await migration
        assert is_blob((await store.documents_table.schema()).field("original"))
        assert await DocumentRepository(store).get_original(doc_id) == body


@pytest.mark.parametrize("failing", ["ensure_indexes", "vacuum"])
async def test_a_migration_stopped_after_the_rewrite_completes_on_rerun(
    temp_db_path, monkeypatch, failing
):
    body = os.urandom(200_000)
    async with Store(temp_db_path, config=_config(), create=True) as store:
        [doc_id] = await at_previous_release(store, [body])
        indexes = sorted(i.name for i in await store.documents_table.list_indices())

    async def fail(*_args, **_kwargs):
        raise RuntimeError("stopped")

    with monkeypatch.context() as patch:
        if failing == "vacuum":
            patch.setattr(Store, "vacuum", fail)
        else:
            patch.setattr(v0_114_0, "ensure_indexes", fail)
        with pytest.raises(RuntimeError, match="stopped"):
            await _migrate(temp_db_path)

    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path, read_only=True):
            pass
    await _migrate(temp_db_path)

    async with Store(temp_db_path, read_only=True) as store:
        table = store.documents_table
        assert sorted(i.name for i in await table.list_indices()) == indexes
        assert len((await table.to_lance()).versions()) == 1
        assert await DocumentRepository(store).get_original(doc_id) == body


async def test_a_skipped_vacuum_fails_the_migration(temp_db_path, monkeypatch):
    async with Store(temp_db_path, config=_config(), create=True) as store:
        await at_previous_release(store, [os.urandom(1_000)])

    async def skipped(*_args, **_kwargs):
        return False

    monkeypatch.setattr(Store, "vacuum", skipped)
    with pytest.raises(RuntimeError, match="migrate' again"):
        await _migrate(temp_db_path)


async def test_a_tag_from_before_reads_and_restores_as_it_was(temp_db_path):
    body = os.urandom(200_000)
    async with Store(temp_db_path, config=_config(), create=True) as store:
        [doc_id] = await at_previous_release(store, [body])
        await store.create_tag("before")

    with capture_logs(v0_114_0.logger, logging.WARNING) as records:
        await _migrate(temp_db_path)
    assert any("before" in r.getMessage() for r in records)

    async with Store(temp_db_path, skip_migration_check=True) as store:
        version = (await store.list_tags())["before"].tables["documents"]
        tagged = lance.dataset((await store.documents_table.to_lance()).uri, version)
        assert tagged.data_storage_version == "2.0"
        assert tagged.to_table(columns=["original"])["original"].to_pylist() == [body]

        await store.restore_tag("before")
        assert (await store.documents_table.schema()).field(
            "original"
        ).type == pa.large_binary()

    await _migrate(temp_db_path)
    async with Store(temp_db_path, read_only=True) as store:
        assert is_blob((await store.documents_table.schema()).field("original"))
        assert await DocumentRepository(store).get_original(doc_id) == body


async def test_the_cloud_rewrite_goes_through_lancedb(temp_db_path, monkeypatch):
    """Runs the Cloud branch on a local table. LanceDB Cloud itself is untested."""
    bodies = [os.urandom(200_000), None]
    async with Store(temp_db_path, config=_config(), create=True) as store:
        ids = await at_previous_release(store, bodies)

    def no_lance(*_args, **_kwargs):
        raise AssertionError("the Cloud rewrite used pylance")

    monkeypatch.setattr(v0_114_0.lance, "write_dataset", no_lance)
    async with Store(
        temp_db_path, config=_config(), skip_migration_check=True
    ) as store:
        monkeypatch.setattr(
            Store, "_connection_mode", property(lambda _: ConnectionMode.CLOUD)
        )
        await store.migrate()
        monkeypatch.undo()

        assert is_blob((await store.documents_table.schema()).field("original"))
        repo = DocumentRepository(store)
        assert [await repo.get_original(i) for i in ids] == bodies
