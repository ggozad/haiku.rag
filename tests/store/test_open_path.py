import lancedb
import pytest

from haiku.rag.store.engine import Store


@pytest.fixture
def counts(monkeypatch):
    """Count the connection-level calls an open makes."""
    tally: dict[str, int] = {"list_tables": 0, "open_settings": 0, "settings_query": 0}

    list_tables = lancedb.AsyncConnection.list_tables
    open_table = lancedb.AsyncConnection.open_table
    query = lancedb.AsyncTable.query

    async def counted_list_tables(self, *args, **kwargs):
        tally["list_tables"] += 1
        return await list_tables(self, *args, **kwargs)

    async def counted_open_table(self, name, *args, **kwargs):
        if name == "settings":
            tally["open_settings"] += 1
        return await open_table(self, name, *args, **kwargs)

    def counted_query(self):
        if self.name == "settings":
            tally["settings_query"] += 1
        return query(self)

    monkeypatch.setattr(lancedb.AsyncConnection, "list_tables", counted_list_tables)
    monkeypatch.setattr(lancedb.AsyncConnection, "open_table", counted_open_table)
    monkeypatch.setattr(lancedb.AsyncTable, "query", counted_query)
    return tally


@pytest.mark.asyncio
async def test_reopening_reads_the_table_list_and_settings_once(temp_db_path, counts):
    async with Store(temp_db_path, create=True):
        pass
    for key in counts:
        counts[key] = 0

    async with Store(temp_db_path):
        pass

    assert counts["list_tables"] == 1
    assert counts["open_settings"] == 1
    assert counts["settings_query"] == 1


@pytest.mark.asyncio
async def test_storage_failures_propagate(temp_db_path):
    """A read failure must not read as empty settings: the migration check would
    then see version 0.0.0 and declare every migration pending."""
    async with Store(temp_db_path, create=True) as store:

        def boom():
            raise RuntimeError("s3 is having a day")

        store.settings_table.query = boom

        with pytest.raises(RuntimeError, match="s3 is having a day"):
            await store._read_stored_settings()


@pytest.mark.asyncio
async def test_non_dict_settings_read_as_empty(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.settings_table.update({"settings": "[]"}, where="id = 'settings'")

        assert await store._read_stored_settings() == {}
