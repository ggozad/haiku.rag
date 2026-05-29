"""haiku-ingester run-batch CLI: echoes the batch report and exits non-zero
when any job dead-letters. IngesterApp is patched so no real ingestion runs."""

from unittest.mock import AsyncMock

from typer.testing import CliRunner

from haiku.rag.ingester.app import BatchReport
from haiku.rag.ingester.cli import _cli as cli

runner = CliRunner()


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
