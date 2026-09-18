from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluations.config import DatasetSpec, DocumentPayload
from evaluations.experiment import config_hash
from evaluations.population import Throughput, _ingest_batched, populate_db
from haiku.rag.config.models import AppConfig


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestThroughput:
    def test_reports_every_n_ingested_documents_with_the_cumulative_rate(self) -> None:
        clock = _Clock()
        throughput = Throughput(total=10, every=2, clock=clock)

        assert throughput.advance(ingested=False) is None
        assert throughput.advance(ingested=True) is None
        clock.now += 60
        line = throughput.advance(ingested=True)

        assert line is not None
        assert "3/10 documents" in line
        assert "2 ingested" in line
        assert "0:01:00" in line
        assert "2.00 documents/min" in line
        assert "ETA 0:03:30" in line

    def test_the_rate_is_over_the_whole_run_not_the_last_window(self) -> None:
        clock = _Clock()
        throughput = Throughput(total=100, every=2, clock=clock)
        throughput.advance(ingested=True)
        clock.now += 60
        throughput.advance(ingested=True)
        throughput.advance(ingested=True)
        clock.now += 240
        line = throughput.advance(ingested=True)
        assert line is not None
        assert "0.80 documents/min" in line

    def test_a_report_with_nothing_ingested_has_no_eta(self) -> None:
        throughput = Throughput(total=5, clock=_Clock())
        line = throughput.report()
        assert "0.00 documents/min" in line
        assert "ETA unknown" in line


def _rag(complete_uris: list[str]) -> MagicMock:
    def _table(rows: list[dict]) -> MagicMock:
        table = MagicMock()
        table.query.return_value.select.return_value.to_list = AsyncMock(
            return_value=rows
        )
        return table

    rag = MagicMock()
    rag.store.document_meta_table = _table(
        [{"id": f"id-{u}", "uri": u} for u in complete_uris]
    )
    rag.store.chunks_table = _table([{"document_id": f"id-{u}"} for u in complete_uris])
    rag.store.vacuum = AsyncMock()
    rag.convert = AsyncMock(side_effect=lambda content, **kw: f"docling:{content}")
    rag.chunk = AsyncMock(return_value=[])
    rag.import_documents = AsyncMock()
    rag.delete_document = AsyncMock()
    return rag


def _spec(batch_size: int | None = None) -> DatasetSpec:
    return DatasetSpec(
        key="test",
        db_filename="test.lancedb",
        document_loader=lambda: [
            {"uri": "u0"},
            {"uri": "u1"},
            {"uri": "bad"},
            {"uri": "u2"},
        ],  # type: ignore[arg-type,return-value]  # ty: ignore[invalid-argument-type]
        document_mapper=lambda doc: (
            None
            if doc["uri"] == "bad"
            else DocumentPayload(uri=doc["uri"], content=f"text {doc['uri']}")
        ),
        qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        ingest_batch_size=batch_size,
    )


class TestIngestReportsWhatItIngested:
    @pytest.mark.asyncio
    async def test_batched_ingest_tells_skipped_from_ingested(self) -> None:
        seen: list[bool] = []
        await _ingest_batched(
            _rag(complete_uris=["u1"]),
            _spec(),
            [{"uri": "u0"}, {"uri": "u1"}, {"uri": "bad"}, {"uri": "u2"}],
            batch_size=10,
            on_document=seen.append,
        )
        assert seen == [True, False, False, True]

    @pytest.mark.asyncio
    async def test_populate_prints_cumulative_throughput(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rag = _rag(complete_uris=["u1"])
        with patch("evaluations.population.HaikuRAG") as haiku:
            haiku.return_value.__aenter__.return_value = rag
            await populate_db(
                _spec(batch_size=10),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                report_every=1,
            )
        out = capsys.readouterr().out
        assert "1/4 documents, 1 ingested" in out
        assert "4/4 documents, 2 ingested" in out
        assert out.count("documents/min") == 3

    @pytest.mark.asyncio
    async def test_population_leaves_the_caller_config_alone(
        self, tmp_path: Path
    ) -> None:
        config = AppConfig()
        before = config_hash(config)
        with patch("evaluations.population.HaikuRAG") as haiku:
            haiku.return_value.__aenter__.return_value = _rag(complete_uris=[])
            await populate_db(
                _spec(batch_size=10), config, db_path=tmp_path / "test.lancedb"
            )

        assert config.storage.auto_vacuum is True
        assert config_hash(config) == before
        assert haiku.call_args.kwargs["config"].storage.auto_vacuum is False
