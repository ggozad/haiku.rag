import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.client.session import SingleDatabaseSession
from haiku.rag.config import AppConfig
from haiku.rag.curate.chunks import chunk_stats, chunk_text_hash
from haiku.rag.curate.store.models import (
    CurrentFingerprint,
    DatabaseSweep,
    NewFingerprint,
    Refresh,
    SweepStatus,
)
from haiku.rag.curate.store.repository import CurateRepository
from haiku.rag.similarity import document_centroids
from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import MigrationRequiredError, SourceUnavailableError
from haiku.rag.utils.sql import escape_sql_string

logger = logging.getLogger(__name__)

# Documents per chunk read.
READ_BATCH = 100


@dataclass(frozen=True)
class _Document:
    id: str
    uri: str | None
    title: str | None
    metadata: dict
    created_at: str | None
    updated_at: str | None
    change_key: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _embedder_label(
    embedding: tuple[str | None, str | None, int | None] | None,
) -> str | None:
    if embedding is None:
        return None
    provider, name, dim = embedding
    return f"{provider}:{name}:{dim}"


def _change_key(md5: str | None, chunk_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps([md5, sorted(chunk_ids)]).encode()).hexdigest()


async def sweep(config: AppConfig, repository: CurateRepository) -> list[DatabaseSweep]:
    """Sweep every configured database once, recording each as it finishes."""
    scope = DatabaseScope.resolve(config)
    if config.curate.databases is not None:
        scope = scope.select(config.curate.databases)
    results = []
    for ref in scope.databases:
        result = await sweep_database(ref, config, repository)
        await repository.record(result)
        results.append(result)
    return results


async def sweep_database(
    ref: DatabaseRef, config: AppConfig, repository: CurateRepository
) -> DatabaseSweep:
    """Read one database and compare it with its current fingerprints; writes nothing."""
    started_at = _now()
    # Every read sees the latest commit, so the version guard sees a write that
    # lands during the sweep.
    strong = config.model_copy(
        update={
            "lancedb": config.lancedb.model_copy(
                update={"read_consistency_interval_seconds": 0}
            )
        }
    )
    session = SingleDatabaseSession(ref, strong, read_only=True, skip_validation=True)
    try:
        await session.open()
    except (SourceUnavailableError, MigrationRequiredError) as error:
        return _failed(ref.name, started_at, str(error))
    try:
        return await _sweep_store(
            session.store, ref.name, config, repository, started_at
        )
    except Exception as error:
        logger.exception("Sweeping database %r failed", ref.name)
        return _failed(ref.name, started_at, f"sweep failed: {type(error).__name__}")
    finally:
        await session.aclose()


def _failed(name: str, started_at: str, error: str) -> DatabaseSweep:
    return DatabaseSweep(
        database=name,
        started_at=started_at,
        finished_at=_now(),
        status=SweepStatus.ERROR,
        error=error,
    )


async def _sweep_store(
    store: Store,
    name: str,
    config: AppConfig,
    repository: CurateRepository,
    started_at: str,
) -> DatabaseSweep:
    versions = await store.current_table_versions()
    embedder = _embedder_label(store.stored_embedding)
    last = await repository.last_ok_sweep(name)
    if last is not None and last.table_versions == versions:
        return DatabaseSweep(
            database=name,
            started_at=started_at,
            finished_at=_now(),
            status=SweepStatus.UNCHANGED,
            table_versions=versions,
            embedder=embedder,
        )

    rebaseline = last is not None and last.embedder != embedder
    current = await repository.current_fingerprints(name)
    documents = await _read_documents(store)
    changed = [
        document
        for document in documents
        if rebaseline
        or document.id not in current
        or current[document.id].change_key != document.change_key
    ]
    new = await _read_changed(store, changed, embedder, config)

    if await store.current_table_versions() != versions:
        return DatabaseSweep(
            database=name,
            started_at=started_at,
            finished_at=_now(),
            status=SweepStatus.MOVED,
            table_versions=versions,
            embedder=embedder,
        )

    present = {document.id for document in documents}
    changed_ids = {document.id for document in changed}
    return DatabaseSweep(
        database=name,
        started_at=started_at,
        finished_at=_now(),
        status=SweepStatus.OK,
        table_versions=versions,
        embedder=embedder,
        rebaseline=rebaseline,
        documents=len(documents),
        new=new,
        replaced=[current[d].id for d in changed_ids if d in current],
        deleted=[fp.id for doc_id, fp in current.items() if doc_id not in present],
        refreshed=_refreshes(documents, current, changed_ids),
    )


def _refreshes(
    documents: list[_Document],
    current: dict[str, CurrentFingerprint],
    changed_ids: set[str],
) -> list[Refresh]:
    return [
        Refresh(
            fingerprint_id=current[document.id].id,
            uri=document.uri,
            title=document.title,
            source_revision=document.metadata.get("source_revision"),
            metadata_keys=sorted(document.metadata),
            updated_at=document.updated_at,
        )
        for document in documents
        if document.id in current and document.id not in changed_ids
    ]


async def _read_documents(store: Store) -> list[_Document]:
    meta_rows = (
        await store.document_meta_table.query()
        .select(["id", "uri", "title", "metadata", "created_at", "updated_at"])
        .to_list()
    )
    chunk_rows = (
        await store.chunks_table.query().select(["id", "document_id"]).to_arrow()
    )
    chunk_ids: dict[str, list[str]] = defaultdict(list)
    for chunk_id, document_id in zip(
        chunk_rows.column("id").to_pylist(),
        chunk_rows.column("document_id").to_pylist(),
        strict=True,
    ):
        chunk_ids[document_id].append(chunk_id)

    documents = []
    for row in meta_rows:
        metadata = json.loads(row.get("metadata") or "{}")
        documents.append(
            _Document(
                id=row["id"],
                uri=row.get("uri"),
                title=row.get("title"),
                metadata=metadata,
                created_at=row.get("created_at") or None,
                updated_at=row.get("updated_at") or None,
                change_key=_change_key(metadata.get("md5"), chunk_ids[row["id"]]),
            )
        )
    return documents


def _id_filter(column: str, ids: list[str]) -> str:
    quoted = ", ".join(f"'{escape_sql_string(i)}'" for i in ids)
    return f"{column} IN ({quoted})"


async def _read_changed(
    store: Store, documents: list[_Document], embedder: str | None, config: AppConfig
) -> list[NewFingerprint]:
    fingerprints = []
    for start in range(0, len(documents), READ_BATCH):
        batch = documents[start : start + READ_BATCH]
        ids = [document.id for document in batch]
        chunks = (
            await store.chunks_table.query()
            .where(_id_filter("document_id", ids))
            .select(["document_id", "content", "vector"])
            .to_arrow()
        )
        dim = chunks.schema.field("vector").type.list_size
        vector_column = chunks.column("vector").combine_chunks()
        vectors = vector_column.values.to_numpy(zero_copy_only=False).reshape(-1, dim)
        embedded = vectors.any(axis=1) if vectors.size else np.zeros(0, dtype=bool)
        centroid_ids, sums, counts = document_centroids(
            chunks.column("document_id"), vectors, embedded, dim
        )
        centroids = {
            doc_id: (sums[i], int(counts[i])) for i, doc_id in enumerate(centroid_ids)
        }
        texts: dict[str, list[str]] = defaultdict(list)
        for document_id, content in zip(
            chunks.column("document_id").to_pylist(),
            chunks.column("content").to_pylist(),
            strict=True,
        ):
            texts[document_id].append(content)

        for document in batch:
            chunk_texts = texts[document.id]
            summed, embedded_count = centroids.get(document.id, (None, 0))
            fingerprints.append(
                NewFingerprint(
                    document_id=document.id,
                    uri=document.uri,
                    title=document.title,
                    change_key=document.change_key,
                    md5=document.metadata.get("md5"),
                    content_type=document.metadata.get("content_type"),
                    source_revision=document.metadata.get("source_revision"),
                    metadata_keys=sorted(document.metadata),
                    chunks=len(chunk_texts),
                    embedded_chunks=embedded_count,
                    chars=sum(len(text) for text in chunk_texts),
                    replacement_chars=sum(text.count("\ufffd") for text in chunk_texts),
                    centroid=_normalized(summed),
                    embedder=embedder,
                    chunk_stats=chunk_stats(
                        [len(text) for text in chunk_texts],
                        config.curate.thresholds.short_chunk_chars,
                    ),
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                    chunk_texts=[(chunk_text_hash(t), len(t)) for t in chunk_texts],
                )
            )
    return fingerprints


def _normalized(summed: np.ndarray | None) -> bytes | None:
    if summed is None:
        return None
    norm = float(np.linalg.norm(summed))
    if norm == 0:
        return None
    return (summed / norm).astype("<f4").tobytes()
