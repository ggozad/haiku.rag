"""Populating an evaluation database from a dataset spec."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.progress import Progress

from evaluations.config import DatasetSpec
from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import DocumentImport
from haiku.rag.config import AppConfig

console = Console()


def _hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


@dataclass
class Throughput:
    """Cumulative ingest rate over the whole build, reported every `every`
    ingested documents. Documents skipped on resume count as seen, not
    ingested, so they never inflate the rate."""

    total: int
    every: int = 50
    clock: Callable[[], float] = time.monotonic
    seen: int = 0
    ingested: int = 0
    started: float = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def advance(self, ingested: bool) -> str | None:
        self.seen += 1
        if not ingested:
            return None
        self.ingested += 1
        return self.report() if self.ingested % self.every == 0 else None

    def report(self) -> str:
        elapsed = self.clock() - self.started
        rate = self.ingested / elapsed * 60 if elapsed > 0 else 0.0
        remaining = self.total - self.seen
        eta = _hms(remaining / rate * 60) if rate > 0 else "unknown"
        return (
            f"{self.seen}/{self.total} documents, {self.ingested} ingested in "
            f"{_hms(elapsed)}: {rate:.2f} documents/min over the whole run, ETA {eta}"
        )


async def _ingest_batched(
    rag: HaikuRAG,
    spec: DatasetSpec,
    corpus,
    batch_size: int,
    on_document: Callable[[bool], None] = lambda ingested: None,
) -> None:
    """Ingest inline-content documents via `import_documents` batches.

    Each batch writes the documents/chunks/document_items tables once and
    embeds every chunk in one batched pass. A URI is skipped on resume only
    when its document has chunks; a chunkless document (crash between the
    document and chunk writes) is deleted and re-imported.
    """
    uri_rows = await (
        rag.store.document_meta_table.query().select(["id", "uri"]).to_list()
    )
    chunk_rows = await rag.store.chunks_table.query().select(["document_id"]).to_list()
    chunked_ids = {row["document_id"] for row in chunk_rows}
    complete = {row["uri"] for row in uri_rows if row["id"] in chunked_ids}
    chunkless = {
        row["uri"]: row["id"] for row in uri_rows if row["id"] not in chunked_ids
    }

    batch: list[DocumentImport] = []
    for doc in corpus:
        payload = spec.document_mapper(cast(Mapping[str, Any], doc))
        if payload is None or payload.uri in complete:
            on_document(False)
            continue
        if payload.uri in chunkless:
            await rag.delete_document(chunkless[payload.uri])
        assert payload.content is not None, "batched ingest requires inline content"
        docling_document = await rag.convert(payload.content, format=payload.format)
        chunks = await rag.chunk(docling_document)
        batch.append(
            DocumentImport(
                docling_document=docling_document,
                chunks=chunks,
                uri=payload.uri,
                title=payload.title,
                metadata=payload.metadata or {},
            )
        )
        if len(batch) >= batch_size:
            await rag.import_documents(batch)
            batch = []
        on_document(True)

    if batch:
        await rag.import_documents(batch)


async def populate_db(
    spec: DatasetSpec,
    config: AppConfig,
    db_path: Path | None = None,
    vacuum_interval: int = 100,
    report_every: int = 50,
) -> None:
    db = spec.db_path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    corpus = spec.document_loader()
    if spec.document_limit is not None:
        corpus = corpus.select(range(min(spec.document_limit, len(corpus))))

    # Disable auto_vacuum - we'll vacuum periodically instead to prevent disk exhaustion
    config.storage.auto_vacuum = False

    throughput = Throughput(total=len(corpus), every=report_every)
    with Progress() as progress:
        task = progress.add_task("[green]Populating database...", total=len(corpus))

        def advance(ingested: bool) -> None:
            progress.advance(task)
            line = throughput.advance(ingested)
            if line is not None:
                console.print(line)

        async with HaikuRAG(db, config=config, create=True) as rag:
            if spec.ingest_batch_size is not None:
                await _ingest_batched(
                    rag,
                    spec,
                    corpus,
                    batch_size=spec.ingest_batch_size,
                    on_document=advance,
                )
                await rag.store.vacuum(retention_seconds=0)
                console.print(throughput.report())
                return

            docs_since_vacuum = 0
            for doc in corpus:
                doc_mapping = cast(Mapping[str, Any], doc)
                payload = spec.document_mapper(doc_mapping)
                if payload is None:
                    advance(False)
                    continue

                # `payload.uri` is the canonical document identifier and is now
                # honored by both `create_document` and (via the `uri=` override)
                # `create_document_from_source`, so it's also the right key to
                # look up an existing document, regardless of whether the source
                # is a file path or inline content.
                existing = await rag.get_document_by_uri(payload.uri)
                if existing is not None:
                    assert existing.id
                    chunks = await rag.chunk_repository.get_by_document_id(existing.id)
                    if chunks:
                        advance(False)
                        continue
                    await rag.document_repository.delete(existing.id)

                if payload.source_path is not None:
                    await rag.create_document_from_source(
                        source=payload.source_path,
                        title=payload.title,
                        metadata=payload.metadata,
                        uri=payload.uri,
                    )
                else:
                    assert payload.content is not None
                    await rag.create_document(
                        content=payload.content,
                        uri=payload.uri,
                        title=payload.title,
                        metadata=payload.metadata,
                        format=payload.format,
                    )
                docs_since_vacuum += 1
                advance(True)

                # Periodic vacuum to prevent disk exhaustion
                if docs_since_vacuum >= vacuum_interval:
                    await rag.store.vacuum(retention_seconds=0)
                    docs_since_vacuum = 0

            # Final vacuum
            await rag.store.vacuum(retention_seconds=0)
    console.print(throughput.report())
