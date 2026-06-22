"""haiku-ingester CLI: exercises every subcommand via CliRunner with
IngesterApp / open_queue patched out so no real ingestion runs."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from haiku.rag.ingester.app import BatchReport
from haiku.rag.ingester.batch import (
    BatchChange,
    BatchDryRunReport,
    BatchManifest,
    BatchSourceSummary,
)
from haiku.rag.ingester.cli import _cli as cli
from haiku.rag.ingester.queue.models import JobOp

runner = CliRunner()


# --- helpers ---


def _fake_app(report: BatchReport, monkeypatch) -> AsyncMock:
    fake = AsyncMock()
    fake.run_batch.return_value = report
    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", lambda **_: fake)
    return fake


def _manifest() -> BatchManifest:
    now = datetime(2026, 6, 22, 10, 30, tzinfo=UTC)
    return BatchManifest(
        generated_at=now,
        sources=[
            BatchSourceSummary(
                source_id="docs",
                upsert_count=1,
                delete_count=1,
                unchanged_count=2,
            )
        ],
        changes=[
            BatchChange(
                op=JobOp.UPSERT,
                source_id="docs",
                uri="file:///a.md",
                revision="r1",
                discovered_at=now,
            ),
            BatchChange(
                op=JobOp.DELETE,
                source_id="docs",
                uri="file:///gone.md",
                discovered_at=now,
            ),
        ],
    )


def _fake_dry_run_app(report: BatchDryRunReport, monkeypatch) -> AsyncMock:
    fake = AsyncMock()
    fake.run_batch_dry_run.return_value = report
    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", lambda **_: fake)
    return fake


def _fake_manifest_app(report: BatchReport, monkeypatch) -> AsyncMock:
    fake = AsyncMock()
    fake.run_batch_from_manifest.return_value = report
    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", lambda **_: fake)
    return fake


def _write_manifest(path) -> None:
    path.write_text(yaml.safe_dump(_manifest().model_dump(mode="json")))


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


def test_run_batch_dry_run_writes_default_manifest(monkeypatch, tmp_path):
    fake = _fake_dry_run_app(
        BatchDryRunReport(manifest=_manifest()),
        monkeypatch,
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["run-batch", "--dry-run", "--db", "x.lancedb"])

        assert result.exit_code == 0
        assert "Dry run complete: 1 upsert, 1 delete, 2 unchanged -> manifest-" in (
            result.output
        )
        written = list(tmp_path.glob("*/manifest-*.yaml"))
        assert len(written) == 1
        data = yaml.safe_load(written[0].read_text())

    assert data["version"] == 1
    assert data["sources"][0]["source_id"] == "docs"
    assert [change["op"] for change in data["changes"]] == ["upsert", "delete"]
    fake.run_batch_dry_run.assert_awaited_once()
    fake.run_batch.assert_not_awaited()


def test_run_batch_dry_run_writes_explicit_output(monkeypatch, tmp_path):
    output = tmp_path / "custom.yaml"
    _fake_dry_run_app(BatchDryRunReport(manifest=_manifest()), monkeypatch)

    result = runner.invoke(
        cli,
        ["run-batch", "--dry-run", "--output", str(output), "--db", "x.lancedb"],
    )

    assert result.exit_code == 0
    assert f"-> {output}" in result.output
    data = yaml.safe_load(output.read_text())
    assert data["changes"][0]["uri"] == "file:///a.md"


def test_run_batch_dry_run_exits_nonzero_when_sweep_fails(monkeypatch, tmp_path):
    output = tmp_path / "failed.yaml"
    _fake_dry_run_app(
        BatchDryRunReport(manifest=_manifest(), failed_sweeps=["docs"]),
        monkeypatch,
    )

    result = runner.invoke(
        cli,
        ["run-batch", "--dry-run", "--output", str(output), "--db", "x.lancedb"],
    )

    assert result.exit_code == 1
    assert "failed to sweep: docs" in result.output
    assert not output.exists()


def test_run_batch_manifest_replays_manifest(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)
    fake = _fake_manifest_app(BatchReport(succeeded=2, dead=0), monkeypatch)

    result = runner.invoke(
        cli, ["run-batch", "--manifest", str(manifest_path), "--db", "x.lancedb"]
    )

    assert result.exit_code == 0
    assert "Manifest batch complete: 2 succeeded, 0 dead" in result.output
    fake.run_batch_from_manifest.assert_awaited_once()
    loaded = fake.run_batch_from_manifest.await_args.args[0]
    assert isinstance(loaded, BatchManifest)
    assert loaded.changes[0].uri == "file:///a.md"
    fake.run_batch.assert_not_awaited()
    fake.run_batch_dry_run.assert_not_awaited()


def test_run_batch_manifest_exits_nonzero_when_dead(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)
    _fake_manifest_app(BatchReport(succeeded=1, dead=1), monkeypatch)

    result = runner.invoke(cli, ["run-batch", "--manifest", str(manifest_path)])

    assert result.exit_code == 1
    assert "1 dead" in result.output


def test_run_batch_manifest_reports_validation_error(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)
    fake = AsyncMock()
    fake.run_batch_from_manifest.side_effect = ValueError("bad manifest")
    monkeypatch.setattr("haiku.rag.ingester.cli.IngesterApp", lambda **_: fake)

    result = runner.invoke(cli, ["run-batch", "--manifest", str(manifest_path)])

    assert result.exit_code == 1
    assert "Error: bad manifest" in result.output


def test_run_batch_manifest_conflicts_with_dry_run(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)

    result = runner.invoke(
        cli, ["run-batch", "--manifest", str(manifest_path), "--dry-run"]
    )

    assert result.exit_code != 0
    assert "--manifest cannot be combined with --dry-run" in result.output


def test_run_batch_manifest_conflicts_with_output(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)

    result = runner.invoke(
        cli,
        [
            "run-batch",
            "--manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "out.yaml"),
        ],
    )

    assert result.exit_code != 0
    assert "--output is only valid with --dry-run" in result.output


def test_run_batch_output_requires_dry_run(tmp_path):
    result = runner.invoke(
        cli,
        ["run-batch", "--output", str(tmp_path / "out.yaml")],
    )

    assert result.exit_code != 0
    assert "--output is only valid with --dry-run" in result.output


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
