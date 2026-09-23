from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from datasets import Dataset

from evaluations.config import DatasetSpec, DocumentPayload
from evaluations.population import Throughput, _ingest_batched, populate_db
from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig
from haiku.rag.store.engine import Store
from haiku.rag.store.repositories import ChunkRepository, DocumentRepository


def _spec(corpus: Dataset, **kwargs: object) -> DatasetSpec:
    defaults: dict[str, object] = {
        "key": "sample",
        "db_filename": "sample.lancedb",
        "document_loader": lambda: corpus,
        "document_mapper": lambda doc: DocumentPayload(
            uri=doc["uri"], content=doc["content"]
        ),
        "qa_loader": lambda: None,
        "qa_case_builder": lambda index, doc: None,
    }
    defaults.update(kwargs)
    return DatasetSpec(**defaults)  # ty: ignore[invalid-argument-type]


def _contexts(rag: MagicMock, progress: MagicMock) -> tuple[MagicMock, MagicMock]:
    haiku = MagicMock()
    haiku.return_value.__aenter__.return_value = rag
    progress_cls = MagicMock()
    progress_cls.return_value.__enter__.return_value = progress
    return haiku, progress_cls


def _rag() -> MagicMock:
    rag = create_autospec(HaikuRAG, instance=True)
    rag.store = create_autospec(Store, instance=True)
    rag.chunk_repository = create_autospec(ChunkRepository, instance=True)
    rag.document_repository = create_autospec(DocumentRepository, instance=True)
    return rag


async def test_populate_batched_limits_corpus_and_vacuums(tmp_path: Path) -> None:
    corpus = Dataset.from_list(
        [
            {"uri": "one", "content": "first"},
            {"uri": "two", "content": "second"},
            {"uri": "three", "content": "third"},
        ]
    )
    spec = _spec(corpus, document_limit=2, ingest_batch_size=10)
    rag = _rag()
    progress = MagicMock()
    haiku, progress_cls = _contexts(rag, progress)

    with (
        patch("evaluations.population.HaikuRAG", haiku),
        patch("evaluations.population.Progress", progress_cls),
        patch(
            "evaluations.population._ingest_batched", new_callable=AsyncMock
        ) as ingest,
    ):
        config = AppConfig()
        await populate_db(spec, config, db_path=tmp_path / "db", vacuum_interval=4)

    assert config.storage.auto_vacuum is True
    assert haiku.call_args.kwargs["config"].storage.auto_vacuum is False
    await_call = ingest.await_args
    assert await_call is not None
    assert len(await_call.args[2]) == 2
    assert await_call.kwargs["batch_size"] == 10
    rag.store.vacuum.assert_awaited_once_with(retention_seconds=0)


async def test_populate_resumes_and_handles_both_document_sources(
    tmp_path: Path,
) -> None:
    corpus = Dataset.from_list(
        [
            {"kind": "ignored", "uri": "ignored"},
            {"kind": "complete", "uri": "complete"},
            {"kind": "chunkless", "uri": "chunkless"},
            {"kind": "file", "uri": "file"},
            {"kind": "inline", "uri": "inline"},
        ]
    )

    def map_document(doc: dict[str, str]) -> DocumentPayload | None:
        if doc["kind"] == "ignored":
            return None
        if doc["kind"] == "file":
            return DocumentPayload(
                uri=doc["uri"], source_path=tmp_path / "source.pdf", title="File"
            )
        return DocumentPayload(
            uri=doc["uri"],
            content=f"content:{doc['uri']}",
            title=doc["kind"],
            metadata={"kind": doc["kind"]},
            format="html",
        )

    spec = _spec(corpus, document_mapper=map_document)
    rag = _rag()
    rag.get_document_by_uri.side_effect = [
        SimpleNamespace(id="complete-id"),
        SimpleNamespace(id="chunkless-id"),
        None,
        None,
    ]
    rag.chunk_repository.get_by_document_id.side_effect = [[object()], []]
    progress = MagicMock()
    haiku, progress_cls = _contexts(rag, progress)

    with (
        patch("evaluations.population.HaikuRAG", haiku),
        patch("evaluations.population.Progress", progress_cls),
    ):
        await populate_db(spec, AppConfig(), db_path=tmp_path / "db", vacuum_interval=2)

    rag.document_repository.delete.assert_awaited_once_with("chunkless-id")
    rag.create_document_from_source.assert_awaited_once_with(
        source=tmp_path / "source.pdf",
        title="File",
        metadata=None,
        uri="file",
    )
    assert [call.kwargs["uri"] for call in rag.create_document.await_args_list] == [
        "chunkless",
        "inline",
    ]
    assert rag.store.vacuum.await_count == 2
    assert progress.advance.call_count == len(corpus)


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


def _resuming_rag(complete_uris: list[str]) -> MagicMock:
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


def _resuming_spec(batch_size: int | None = None) -> DatasetSpec:
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
    async def test_batched_ingest_tells_skipped_from_ingested(self) -> None:
        seen: list[bool] = []
        await _ingest_batched(
            _resuming_rag(complete_uris=["u1"]),
            _resuming_spec(),
            [{"uri": "u0"}, {"uri": "u1"}, {"uri": "bad"}, {"uri": "u2"}],
            batch_size=10,
            on_document=seen.append,
        )
        assert seen == [False, False, True, True]

    async def test_a_document_counts_as_ingested_once_its_batch_is_written(
        self,
    ) -> None:
        rag = _resuming_rag(complete_uris=[])
        rag.import_documents = AsyncMock(side_effect=[None, RuntimeError("disk full")])
        seen: list[bool] = []
        with pytest.raises(RuntimeError, match="disk full"):
            await _ingest_batched(
                rag,
                _resuming_spec(),
                [{"uri": "u0"}, {"uri": "u1"}, {"uri": "u2"}, {"uri": "u3"}],
                batch_size=2,
                on_document=seen.append,
            )
        assert seen == [True, True]

    async def test_populate_prints_cumulative_throughput(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rag = _resuming_rag(complete_uris=["u1"])
        with patch("evaluations.population.HaikuRAG") as haiku:
            haiku.return_value.__aenter__.return_value = rag
            await populate_db(
                _resuming_spec(batch_size=10),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                report_every=1,
            )
        out = capsys.readouterr().out
        assert "3/4 documents, 1 ingested" in out
        assert "4/4 documents, 2 ingested" in out
        assert out.count("documents/min") == 3
