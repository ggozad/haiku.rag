import asyncio
import sys

import pytest
import sqlalchemy as sa
import yaml
from typer.testing import CliRunner

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import CurateStoreConfig
from haiku.rag.curate.app import APIServerStopped
from haiku.rag.curate.cli import _cli as cli
from haiku.rag.curate.cli import cli as curate_cli
from haiku.rag.curate.store.db import SCHEMA_VERSION, schema_version
from haiku.rag.curate.store.migrations import open_store
from tests.curate.test_sweep import _import, _writer_config

runner = CliRunner()


def _write_config(tmp_path, databases: dict) -> str:
    path = tmp_path / "curate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "lancedb": {"databases": {k: str(v) for k, v in databases.items()}},
                "curate": {"store": {"path": str(tmp_path / "curate.db")}},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


async def _create(path, with_document: bool) -> None:
    async with HaikuRAG(path, _writer_config(), create=True) as rag:
        if with_document:
            await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])


def test_sweep_reports_each_database(tmp_path):
    wiki = tmp_path / "wiki.lancedb"
    asyncio.run(_create(wiki, with_document=True))
    config = _write_config(tmp_path, {"wiki": wiki})

    first = runner.invoke(cli, ["--config", config, "sweep"])
    second = runner.invoke(cli, ["--config", config, "sweep"])

    assert first.exit_code == 0, first.output
    assert "wiki: ok, 1 documents, 1 new, 0 deleted" in first.output
    assert second.exit_code == 0
    assert "wiki: unchanged" in second.output


def test_sweep_exits_non_zero_when_a_database_fails(tmp_path):
    wiki = tmp_path / "wiki.lancedb"
    asyncio.run(_create(wiki, with_document=False))
    config = _write_config(tmp_path, {"wiki": wiki, "gone": tmp_path / "gone.lancedb"})

    result = runner.invoke(cli, ["--config", config, "sweep"])

    assert result.exit_code == 1
    assert "wiki: ok" in result.output
    assert "gone: error: database 'gone' does not exist" in result.output


def test_sweep_of_an_unknown_selected_database_is_an_error(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "curate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "lancedb": {"databases": {"wiki": str(tmp_path / "wiki.lancedb")}},
                "curate": {
                    "databases": ["papers"],
                    "store": {"path": str(tmp_path / "curate.db")},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys, "argv", ["haiku-curate", "--config", str(config_path), "sweep"]
    )

    with pytest.raises(SystemExit) as exit_info:
        curate_cli()

    assert exit_info.value.code == 1
    assert "Error: unknown database(s) papers" in capsys.readouterr().err


def test_store_init_and_migrate(tmp_path):
    config = _write_config(tmp_path, {})
    store = tmp_path / "other.db"

    init = runner.invoke(
        cli, ["--config", config, "store", "init", "--store", str(store)]
    )
    migrate = runner.invoke(
        cli, ["--config", config, "store", "migrate", "--store", str(store)]
    )

    assert init.exit_code == 0
    assert f"Store initialized at {store}" in init.output
    assert migrate.exit_code == 0
    assert f"Store at {store} is up to date" in migrate.output
    assert store.exists()


def test_store_init_uses_the_configured_path(tmp_path):
    config = _write_config(tmp_path, {})

    result = runner.invoke(cli, ["--config", config, "store", "init"])

    assert result.exit_code == 0
    assert (tmp_path / "curate.db").exists()


async def _store_from_newer_code(path) -> None:
    engine = await open_store(CurateStoreConfig(path=path))
    async with engine.begin() as conn:
        await conn.execute(sa.update(schema_version).values(version=SCHEMA_VERSION + 1))
    await engine.dispose()


def test_store_from_newer_code_is_a_clean_error(tmp_path, monkeypatch, capsys):
    config = _write_config(tmp_path, {})
    asyncio.run(_store_from_newer_code(tmp_path / "curate.db"))
    monkeypatch.setattr(sys, "argv", ["haiku-curate", "--config", config, "sweep"])

    with pytest.raises(SystemExit) as exit_info:
        curate_cli()

    assert exit_info.value.code == 1
    assert "Error: store schema" in capsys.readouterr().err


@pytest.mark.parametrize("args,sweeping", [([], True), (["--no-sweep"], False)])
def test_serve_passes_the_sweep_choice(tmp_path, monkeypatch, args, sweeping):
    config = _write_config(tmp_path, {})
    calls = []

    async def serve(app_config, *, sweeping):
        calls.append((app_config.curate.store.path, sweeping))

    monkeypatch.setattr("haiku.rag.curate.cli.serve", serve)

    result = runner.invoke(cli, ["--config", config, "serve", *args])

    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path / "curate.db", sweeping)]


def test_a_stopped_api_server_is_a_clean_error(tmp_path, monkeypatch, capsys):
    config = _write_config(tmp_path, {})

    async def serve(app_config, *, sweeping):
        raise APIServerStopped("the API server stopped")

    monkeypatch.setattr("haiku.rag.curate.cli.serve", serve)
    monkeypatch.setattr(sys, "argv", ["haiku-curate", "--config", config, "serve"])

    with pytest.raises(SystemExit) as exit_info:
        curate_cli()

    assert exit_info.value.code == 1
    assert "Error: the API server stopped" in capsys.readouterr().err
