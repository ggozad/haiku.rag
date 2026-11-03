"""The ingester following the folder register: a folder added to it is
watched and ingested, a removed one stops and its documents leave."""

from collections import defaultdict

import pytest
import yaml

from haiku.rag.config import AppConfig, load_yaml_config
from haiku.rag.folders import FolderRegister
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.store.models.document import Document
from tests.ingester.conftest import _ingested, _mock_client, _serving, _wait_until


@pytest.fixture
def clients(use_client):
    clients = defaultdict(_mock_client)
    use_client(clients)
    return clients


def _setup(tmp_path, *folders: str) -> tuple:
    for name in folders:
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.md").write_text(name)
    path = tmp_path / "haiku.rag.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "lancedb": {"databases": {"files": str(tmp_path / "files.lancedb")}},
                "ingester": {
                    "folders_register": "folders.yaml",
                    "queue": {"path": str(tmp_path / "queue.db")},
                    "workers": {
                        "worker_count": 2,
                        "poll_idle_interval_s": 0.05,
                        "retry": {"max_attempts": 1},
                        # What `_serving` sets on the running configuration.
                        "shutdown_grace_s": 0.2,
                    },
                    "api": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    config = AppConfig.model_validate(load_yaml_config(path))
    return path, config, FolderRegister.of(config)


async def test_serving_follows_the_register(tmp_path, clients):
    path, config, register = _setup(tmp_path, "books", "notes")
    register.add(tmp_path / "books")
    config = AppConfig.model_validate(load_yaml_config(path))
    app = IngesterApp(config=config, config_path=path)

    async with _serving(app):
        await _wait_until(lambda: _ingested(clients["files"]) == ["books.md"])

        register.add(tmp_path / "notes")
        await _wait_until(
            lambda: sorted(app._source_databases) == ["books", "notes"], timeout=15
        )
        # Re-adding the database revisits books.md, which the real client
        # skips on its unchanged revision.
        await _wait_until(
            lambda: set(_ingested(clients["files"])) == {"books.md", "notes.md"}
        )

        register.remove("books")
        await _wait_until(
            lambda: sorted(app._source_databases) == ["notes"], timeout=15
        )
        await _wait_until(
            lambda: (
                [
                    call.kwargs
                    for call in clients["files"].delete_document.await_args_list
                ]
                == [{"source_id": "books"}]
            )
        )


def _documents(*owners: str | None) -> list[Document]:
    return [
        Document(
            id=f"d{n}",
            content="x",
            uri=f"file:///d{n}",
            metadata={} if owner is None else {"source_id": owner},
        )
        for n, owner in enumerate(owners, 1)
    ]


async def test_a_retired_folder_loses_its_documents_at_start(tmp_path, clients):
    path, config, register = _setup(tmp_path, "books", "gone")
    client = clients["files"]
    client.list_documents.side_effect = None
    client.list_documents.return_value = _documents("gone", "gone", "books", None)
    register.add(tmp_path / "books")
    register.add(tmp_path / "gone")
    register.remove("gone")
    config = AppConfig.model_validate(load_yaml_config(path))

    report = await IngesterApp(config=config).run_batch()

    assert report.dead == 0
    assert [
        (call.args, call.kwargs) for call in client.delete_document.await_args_list
    ] == [(("d1",), {"source_id": "gone"}), (("d2",), {"source_id": "gone"})]


async def test_a_retired_folder_with_no_documents_deletes_nothing(tmp_path, clients):
    path, config, register = _setup(tmp_path, "books", "gone")
    register.add(tmp_path / "books")
    register.add(tmp_path / "gone")
    register.remove("gone")
    config = AppConfig.model_validate(load_yaml_config(path))

    await IngesterApp(config=config).run_batch()

    assert clients["files"].delete_document.await_count == 0


async def test_a_database_with_only_retired_folders_starts_to_purge(tmp_path, clients):
    path, config, register = _setup(tmp_path, "books")
    client = clients["files"]
    client.list_documents.side_effect = None
    client.list_documents.return_value = _documents("books")
    register.add(tmp_path / "books")
    register.remove("books")
    config = AppConfig.model_validate(load_yaml_config(path))
    app = IngesterApp(config=config, config_path=path)

    async with _serving(app):
        await _wait_until(lambda: client.delete_document.await_count == 1)
    assert client.delete_document.await_args.args == ("d1",)
