from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

from datasets import Dataset

from evaluations.config import DatasetSpec, DocumentPayload
from evaluations.population import populate_db
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

    assert config.storage.auto_vacuum is False
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
