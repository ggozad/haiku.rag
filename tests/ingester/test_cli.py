"""haiku-ingester CLI: exercises every subcommand via CliRunner with
IngesterApp / open_queue patched out so no real ingestion runs."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from haiku.rag.ingester.app import BatchReport
from haiku.rag.ingester.cli import _cli as cli

runner = CliRunner()


# --- helpers ---


def _fake_app(report: BatchReport, monkeypatch) -> AsyncMock:
    fake = AsyncMock()
    fake.run_batch.return_value = report
    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", lambda **_: fake)
    return fake


def test_run_batch_reports_and_exits_zero(monkeypatch):
    fake = _fake_app(BatchReport(succeeded=3, dead=0), monkeypatch)

    result = runner.invoke(cli, ["run-batch", "--db", "x.lancedb"])

    assert result.exit_code == 0
    assert "3 succeeded, 0 dead" in result.output
    fake.run_batch.assert_awaited_once()


def test_run_batch_exits_nonzero_when_dead(monkeypatch):
    _fake_app(BatchReport(succeeded=1, dead=2), monkeypatch)

    result = runner.invoke(cli, ["run-batch", "--db", "x.lancedb"])

    assert result.exit_code == 1
    assert "2 dead" in result.output


def test_run_batch_exits_nonzero_when_sweep_fails(monkeypatch):
    _fake_app(BatchReport(succeeded=2, dead=0, failed_sweeps=["docs"]), monkeypatch)

    result = runner.invoke(cli, ["run-batch", "--db", "x.lancedb"])

    assert result.exit_code == 1
    assert "failed to sweep: docs" in result.output


# --- serve ---


def test_serve_invokes_app(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", lambda **_: fake)

    result = runner.invoke(cli, ["serve", "--db", "x.lancedb", "--no-api"])

    assert result.exit_code == 0
    fake.serve.assert_awaited_once_with(api=False)


def test_serve_passes_host_and_port(monkeypatch):
    fake = AsyncMock()
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", _capture)

    result = runner.invoke(
        cli, ["serve", "--db", "x.lancedb", "--host", "0.0.0.0", "--port", "9999"]
    )

    assert result.exit_code == 0
    assert captured["config"].ingester.api.host == "0.0.0.0"
    assert captured["config"].ingester.api.port == 9999


def test_serve_passes_root_path(monkeypatch):
    fake = AsyncMock()
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", _capture)

    result = runner.invoke(
        cli, ["serve", "--db", "x.lancedb", "--root-path", "/ingester/"]
    )

    assert result.exit_code == 0
    # Assignment is normalized (trailing slash stripped) via validate_assignment.
    assert captured["config"].ingester.api.root_path == "/ingester"


# --- queue init / migrate ---


def test_queue_init(tmp_path, monkeypatch):
    fake_conn = AsyncMock()
    monkeypatch.setattr(
        "haiku.rag.ingester.cli.open_queue", AsyncMock(return_value=fake_conn)
    )

    db_path = tmp_path / "queue.db"
    result = runner.invoke(cli, ["queue", "init", "--queue", str(db_path)])

    assert result.exit_code == 0
    assert "initialized" in result.output


def test_queue_migrate(tmp_path, monkeypatch):
    fake_conn = AsyncMock()
    monkeypatch.setattr(
        "haiku.rag.ingester.cli.open_queue", AsyncMock(return_value=fake_conn)
    )

    db_path = tmp_path / "queue.db"
    result = runner.invoke(cli, ["queue", "migrate", "--queue", str(db_path)])

    assert result.exit_code == 0
    assert "up to date" in result.output


def test_queue_target_masks_dburi_password():
    from haiku.rag.config import QueueConfig
    from haiku.rag.ingester.cli import _queue_target

    target = _queue_target(
        QueueConfig(dburi="postgresql+asyncpg://user:secret@host:5432/db")
    )
    assert "secret" not in target
    assert "***" in target
    assert "user" in target and "host:5432/db" in target


# --- config loading ---


def test_load_config_with_explicit_path(tmp_path):
    from haiku.rag.ingester.cli import _load_config_with_override

    config_file = tmp_path / "test.yaml"
    config_file.write_text("embeddings:\n  model:\n    provider: ollama\n")

    config = _load_config_with_override(config_file)
    assert config.embeddings.model.provider == "ollama"


def test_load_config_falls_back_to_default(monkeypatch):
    from haiku.rag.ingester.cli import _load_config_with_override, get_config

    monkeypatch.setattr("haiku.rag.ingester.cli.find_config_file", lambda _: None)

    config = _load_config_with_override(None)
    assert config == get_config()


# --- cli() entry point ---


def test_cli_entry_point(monkeypatch):
    from haiku.rag.ingester.cli import cli as cli_entry

    mock_cli = MagicMock()
    monkeypatch.setattr("haiku.rag.ingester.cli._cli", mock_cli)
    monkeypatch.setattr("haiku.rag.ingester.cli.configure_cli_logging", lambda: None)
    monkeypatch.setattr("haiku.rag.telemetry.configure", lambda **_: None)

    cli_entry()

    mock_cli.assert_called_once()


def test_cli_entry_point_exits_on_migration_error(monkeypatch):
    from haiku.rag.ingester.cli import cli as cli_entry
    from haiku.rag.store.exceptions import MigrationRequiredError

    monkeypatch.setattr(
        "haiku.rag.ingester.cli._cli",
        MagicMock(side_effect=MigrationRequiredError("need migration")),
    )
    monkeypatch.setattr("haiku.rag.ingester.cli.configure_cli_logging", lambda: None)
    monkeypatch.setattr("haiku.rag.telemetry.configure", lambda **_: None)

    with pytest.raises(SystemExit) as exc_info:
        cli_entry()
    assert exc_info.value.code == 1
