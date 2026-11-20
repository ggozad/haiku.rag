import hashlib
import json

import lancedb
import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.models import Document
from haiku.rag.store.models.recording import RecordingFile
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.recording import RecordingRepository
from haiku.rag.store.schema import REQUIRED_TABLES

VIDEO = RecordingFile(content=b"\x00\x01video bytes" * 1000, suffix=".mp4")


async def _stored(store: Store, content: str = "a recording") -> str:
    """The id of a new document."""
    doc = await DocumentRepository(store).create(Document(content=content))
    assert doc.id is not None
    return doc.id


def _table(store: Store) -> lancedb.AsyncTable:
    assert store.recordings_table is not None
    return store.recordings_table


async def _ids(store: Store) -> list[str]:
    rows = await _table(store).query().select(["document_id"]).to_list()
    return sorted(r["document_id"] for r in rows)


async def test_a_new_database_has_a_recordings_table(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        assert "recordings" in REQUIRED_TABLES
        assert "recordings" in store._tables()
        assert await _ids(store) == []


async def test_a_stored_recording_reads_back_by_range(temp_db_path):
    subtitled = RecordingFile(
        content=VIDEO.content, suffix=".mp4", subtitles="WEBVTT\n"
    )
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, subtitled, duration=12.5)

        stored = await RecordingRepository(store).open(doc)

        assert stored is not None
        assert (stored.suffix, stored.duration, stored.subtitles) == (
            ".mp4",
            12.5,
            "WEBVTT\n",
        )
        assert stored.sha256 == hashlib.sha256(VIDEO.content).hexdigest()
        assert stored.size == len(VIDEO.content)
        assert stored.read_range(2, 11) == VIDEO.content[2:13]


async def test_a_document_without_a_recording_opens_none(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)

        assert await RecordingRepository(store).open(doc) is None


async def test_putting_a_recording_again_replaces_it(temp_db_path):
    other = RecordingFile(content=b"other", suffix=".webm")
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        recordings = RecordingRepository(store)
        await recordings.put(doc, VIDEO, duration=1.0)
        await recordings.put(doc, other, duration=2.0)

        stored = await recordings.open(doc)

        assert await _ids(store) == [doc]
        assert stored is not None
        assert (stored.suffix, stored.size) == (".webm", 5)


async def test_deleting_a_document_deletes_its_recording(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        kept, deleted = await _stored(store, "kept"), await _stored(store, "deleted")
        for doc in (kept, deleted):
            await RecordingRepository(store).put(doc, VIDEO, duration=1.0)

        await DocumentRepository(store).delete(deleted)

        assert await _ids(store) == [kept]


async def test_deleting_every_document_empties_the_recordings(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, VIDEO, duration=1.0)

        await DocumentRepository(store).delete_all()

        assert await _ids(store) == []


async def test_a_failed_replacement_keeps_the_stored_recording(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, VIDEO, duration=1.0)

        with pytest.raises(RuntimeError, match="boom"):
            async with store.write_transaction():
                await RecordingRepository(store).put(
                    doc, RecordingFile(content=b"new", suffix=".mp4"), duration=2.0
                )
                raise RuntimeError("boom")

        stored = await RecordingRepository(store).open(doc)
        assert stored is not None
        assert stored.read_range(0, 32) == VIDEO.content[:32]


async def test_a_failed_write_transaction_rolls_back_the_recording(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)

        with pytest.raises(RuntimeError, match="boom"):
            async with store.write_transaction():
                await RecordingRepository(store).put(doc, VIDEO, duration=1.0)
                raise RuntimeError("boom")

        assert await _ids(store) == []


async def test_a_tag_restores_the_recordings_it_saw(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, VIDEO, duration=1.0)
        await store.create_tag("with-recording")
        await RecordingRepository(store).delete(doc)

        await store.restore_tag("with-recording")

        stored = await RecordingRepository(store).open(doc)
        assert stored is not None
        assert stored.read_range(0, 4) == VIDEO.content[:4]


async def _old_tag(store: Store, name: str) -> None:
    """A tag as a release before the recordings table would have left it."""
    await store.settings_table.update(
        updates={
            "settings": json.dumps({**store.stored_settings, "version": "0.112.0"})
        },
        where="id = 'settings'",
    )
    for table_name, table in store._tables().items():
        if table_name != "recordings":
            await table.tags.create(name, await table.version())
    await store.settings_table.update(
        updates={"settings": json.dumps(store.stored_settings)}, where="id = 'settings'"
    )


async def test_a_tag_from_before_the_recordings_table_is_complete(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await _old_tag(store, "old")

        info = (await store.list_tags())["old"]

        assert info.complete
        assert "recordings" not in info.tables


async def test_a_current_tag_missing_the_recordings_table_is_partial(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        for table_name, table in store._tables().items():
            if table_name != "recordings":
                await table.tags.create("broken", await table.version())

        info = (await store.list_tags())["broken"]

        assert info.missing_tables == ["recordings"]


async def test_restoring_an_old_tag_empties_the_recordings_and_its_safety_tag_brings_them_back(
    temp_db_path,
):
    async with Store(temp_db_path, create=True) as store:
        await _old_tag(store, "old")
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, VIDEO, duration=1.0)

        safety = await store.restore_tag("old")

        assert await _ids(store) == []
        await store.restore_tag(safety)
        assert await _ids(store) == [doc]


async def test_vacuum_keeps_one_fragment_per_recording_and_removes_old_versions(
    temp_db_path, monkeypatch
):
    async with Store(temp_db_path, create=True) as store:
        monkeypatch.setattr(store._config.storage, "auto_vacuum", False)
        for content in ("one", "two"):
            doc = await _stored(store, content)
            await RecordingRepository(store).put(doc, VIDEO, duration=1.0)
        before = len((await _table(store).to_lance()).get_fragments())

        assert await store.vacuum(retention_seconds=0)

        dataset = await _table(store).to_lance()
        assert len(dataset.get_fragments()) == before
        assert len(dataset.versions()) == 1
        stored = await RecordingRepository(store).open(doc)
        assert stored is not None
        assert stored.read_range(0, 32) == VIDEO.content[:32]


async def test_vacuum_keeps_the_recording_a_tag_holds(temp_db_path, monkeypatch):
    async with Store(temp_db_path, create=True) as store:
        monkeypatch.setattr(store._config.storage, "auto_vacuum", False)
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, VIDEO, duration=1.0)
        await store.create_tag("kept")
        await RecordingRepository(store).delete(doc)

        assert await store.vacuum(retention_seconds=0)
        await store.restore_tag("kept")

        stored = await RecordingRepository(store).open(doc)
        assert stored is not None
        assert stored.read_range(0, 32) == VIDEO.content[:32]


async def test_a_database_read_without_the_recordings_table_has_no_recordings(
    temp_db_path,
):
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
    db = await lancedb.connect_async(temp_db_path)
    await db.drop_table("recordings")
    db.close()

    async with Store(temp_db_path, read_only=True, skip_migration_check=True) as store:
        recordings = RecordingRepository(store)
        assert await recordings.open(doc) is None
        assert await recordings.digest(doc) is None


async def test_lancedb_cloud_refuses_to_store_a_recording(temp_db_path, monkeypatch):
    from haiku.rag.store.engine import ConnectionMode
    from haiku.rag.store.exceptions import UnsupportedRecordingStorageError

    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        monkeypatch.setattr(Store, "_connection_mode", ConnectionMode.CLOUD)

        with pytest.raises(UnsupportedRecordingStorageError, match="LanceDB Cloud"):
            await RecordingRepository(store).put(doc, VIDEO, duration=1.0)


async def test_info_at_a_tag_from_before_the_recordings_table_asks_for_a_migration(
    temp_db_path,
):
    from haiku.rag.store.exceptions import MigrationRequiredError
    from haiku.rag.store.info import gather_database_info

    async with Store(temp_db_path, create=True) as store:
        await _old_tag(store, "old")
        config = store._config

    with pytest.raises(MigrationRequiredError, match="restore it in a copy"):
        await gather_database_info(temp_db_path, config, tag="old")


async def test_info_counts_the_stored_files_in_the_recordings_size(temp_db_path):
    from haiku.rag.store.info import gather_database_info

    large = RecordingFile(content=b"\x00" * 4_000_000, suffix=".mp4")
    async with Store(temp_db_path, create=True) as store:
        await RecordingRepository(store).put(await _stored(store), large, duration=1.0)
        config = store._config

    info = await gather_database_info(temp_db_path, config)

    [recordings] = [t for t in info.tables if t.name == "recordings"]
    assert recordings.total_bytes >= len(large.content)


async def test_info_at_a_tag_counts_the_files_the_tag_holds(temp_db_path):
    from haiku.rag.store.info import gather_database_info

    large = RecordingFile(content=b"\x00" * 4_000_000, suffix=".mp4")
    async with Store(temp_db_path, create=True) as store:
        doc = await _stored(store)
        await RecordingRepository(store).put(doc, large, duration=1.0)
        await store.create_tag("kept")
        await RecordingRepository(store).delete(doc)
        config = store._config

    live = await gather_database_info(temp_db_path, config)
    tagged = await gather_database_info(temp_db_path, config, tag="kept")

    def size(info):
        return next(t.total_bytes for t in info.tables if t.name == "recordings")

    assert size(live) < len(large.content) <= size(tagged)


async def test_info_counts_an_inline_recording_once(temp_db_path):
    from haiku.rag.store.info import gather_database_info

    async with Store(temp_db_path, create=True) as store:
        await RecordingRepository(store).put(await _stored(store), VIDEO, duration=1.0)
        [descriptor] = await _table(store).query().select(["content"]).to_list()
        assert descriptor["content"]["kind"] == 0
        stats: dict = await _table(store).stats()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        table_bytes = stats["total_bytes"]
        config = store._config

    info = await gather_database_info(temp_db_path, config)

    [recordings] = [t for t in info.tables if t.name == "recordings"]
    assert recordings.total_bytes == table_bytes
