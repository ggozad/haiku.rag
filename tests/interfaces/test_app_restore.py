import json

import lancedb
import pytest
from rich.console import Console

from haiku.rag.app import HaikuRAGApp
from haiku.rag.config.models import AppConfig
from haiku.rag.store import ReadOnlyError
from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import MigrationRequiredError
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from tests.conftest import for_path

OLD_VERSION = "0.64.0"


def _app(path, read_only: bool = False) -> HaikuRAGApp:
    application = HaikuRAGApp(
        scope=for_path(path), config=AppConfig(), read_only=read_only
    )
    application.console = Console(record=True, width=200)
    return application


def _open(path) -> Store:
    return Store(
        path,
        config=AppConfig(),
        skip_validation=True,
        skip_migration_check=True,
        read_only=True,
    )


async def _set_live_settings(path, **changes) -> None:
    async with Store(
        path, config=AppConfig(), skip_validation=True, skip_migration_check=True
    ) as store:
        settings = {**store.stored_settings, **changes}
        await store.settings_table.update({"settings": json.dumps(settings)})


async def _versions(path) -> dict[str, int]:
    async with _open(path) as store:
        return await store.current_table_versions()


async def _state(path) -> tuple[str, set[str]]:
    async with _open(path) as store:
        docs = await DocumentRepository(store).list_all(include_content=True)
        return await store.get_haiku_version(), {d.content for d in docs}


async def _db_tagged_at_head(path) -> str:
    """A database with 'head' on its first document and a second written after."""
    async with Store(path, config=AppConfig(), create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First"))
        await store.create_tag("head")
        await repo.create(Document(content="Second"))
        return await store.get_haiku_version()


async def test_restore_moves_a_current_database_to_an_older_tag(temp_db_path):
    async with Store(temp_db_path, config=AppConfig(), create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First"))
        current_version = await store.get_haiku_version()
        await store.set_haiku_version(OLD_VERSION)
        await store.create_tag("old")
        await store.set_haiku_version(current_version)
        await repo.create(Document(content="Second"))
    app = _app(temp_db_path)

    await app.restore_tag("old")

    assert await _state(temp_db_path) == (OLD_VERSION, {"First"})
    printed = app.console.export_text()
    assert "Restored database to tag 'old'" in printed
    assert "The previous state is preserved as 'before-restore-" in printed


@pytest.mark.parametrize("live_version", [OLD_VERSION, "0.63.0"])
async def test_restore_returns_to_a_current_tag_over_a_state_needing_migration(
    temp_db_path, live_version
):
    current_version = await _db_tagged_at_head(temp_db_path)
    await _set_live_settings(temp_db_path, version=live_version)
    replaced = await _versions(temp_db_path)

    await _app(temp_db_path).restore_tag("head")

    assert await _state(temp_db_path) == (current_version, {"First"})
    async with _open(temp_db_path) as store:
        tags = await store.list_tags()
    (safety,) = set(tags) - {"head"}
    assert tags[safety].complete
    assert tags[safety].tables == replaced


async def test_restore_returns_to_a_current_tag_over_another_embedder(temp_db_path):
    current_version = await _db_tagged_at_head(temp_db_path)
    async with _open(temp_db_path) as store:
        embeddings = store.stored_settings["embeddings"]
    embeddings["model"]["name"] = "another-embedder"
    await _set_live_settings(temp_db_path, embeddings=embeddings)

    await _app(temp_db_path).restore_tag("head")

    assert await _state(temp_db_path) == (current_version, {"First"})


async def test_restore_in_read_only_mode_changes_nothing(temp_db_path):
    await _db_tagged_at_head(temp_db_path)
    await _set_live_settings(temp_db_path, version=OLD_VERSION)
    before = await _versions(temp_db_path)

    with pytest.raises(ReadOnlyError):
        await _app(temp_db_path, read_only=True).restore_tag("head")

    assert await _versions(temp_db_path) == before


async def test_restore_of_an_old_tag_over_a_state_needing_migration_is_refused(
    temp_db_path,
):
    async with Store(temp_db_path, config=AppConfig(), create=True) as store:
        await store.set_haiku_version(OLD_VERSION)
        await store.create_tag("old")
    before = await _versions(temp_db_path)

    with pytest.raises(MigrationRequiredError):
        await _app(temp_db_path).restore_tag("old")

    assert await _versions(temp_db_path) == before


async def test_restore_names_the_tables_a_database_is_missing(temp_db_path):
    await _db_tagged_at_head(temp_db_path)
    db = await lancedb.connect_async(str(temp_db_path))
    await db.drop_table("document_meta")

    with pytest.raises(ValueError, match="document_meta"):
        await _app(temp_db_path).restore_tag("head")

    assert "document_meta" not in (await db.list_tables()).tables


async def test_restore_of_a_tag_below_the_floor_is_refused(temp_db_path):
    async with Store(temp_db_path, config=AppConfig(), create=True) as store:
        current_version = await store.get_haiku_version()
        await store.set_haiku_version("0.63.0")
        await store.create_tag("ancient")
        await store.set_haiku_version(current_version)
    before = await _versions(temp_db_path)

    with pytest.raises(MigrationRequiredError, match="haiku.rag==0.111.1"):
        await _app(temp_db_path).restore_tag("ancient")

    assert await _versions(temp_db_path) == before


async def test_migrate_below_the_floor_writes_nothing(temp_db_path):
    async with Store(temp_db_path, config=AppConfig(), create=True) as store:
        await store.set_haiku_version("0.57.0")
        await store.db.drop_table("document_meta")
    db = await lancedb.connect_async(str(temp_db_path))
    tables = set((await db.list_tables()).tables)
    before = {name: await (await db.open_table(name)).version() for name in tables}

    with pytest.raises(MigrationRequiredError, match="haiku.rag==0.111.1"):
        await _app(temp_db_path).migrate()

    assert set((await db.list_tables()).tables) == tables
    assert {
        name: await (await db.open_table(name)).version() for name in tables
    } == before


async def test_migrate_at_the_floor_applies_the_kept_steps(temp_db_path):
    async with Store(temp_db_path, config=AppConfig(), create=True) as store:
        current_version = await store.get_haiku_version()
        await store.set_haiku_version("0.64.0")

    assert await _app(temp_db_path).migrate()

    assert (await _state(temp_db_path))[0] == current_version


async def test_migrate_of_a_database_without_settings_initializes_it(temp_db_path):
    temp_db_path.mkdir()

    assert await _app(temp_db_path).migrate() == []

    async with _open(temp_db_path) as store:
        assert "settings" in (await store.db.list_tables()).tables


async def test_migrate_of_a_missing_path_creates_nothing(temp_db_path):
    with pytest.raises(FileNotFoundError):
        await _app(temp_db_path).migrate()

    assert not temp_db_path.exists()
