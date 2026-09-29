import json

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from haiku.rag.curate.store.db import chunk_texts, fingerprints, sweeps
from haiku.rag.curate.store.models import (
    CurrentFingerprint,
    DatabaseSweep,
    Fingerprint,
    LastSweep,
    SweepStatus,
)


class CurateRepository:
    """Reads and writes of curate's store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def last_ok_sweep(self, database: str) -> LastSweep | None:
        stmt = (
            sa.select(sweeps.c.id, sweeps.c.table_versions, sweeps.c.embedder)
            .where(sweeps.c.database == database, sweeps.c.status == SweepStatus.OK)
            .order_by(sweeps.c.id.desc())
            .limit(1)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        if row is None:
            return None
        return LastSweep(
            id=row.id,
            table_versions=json.loads(row.table_versions),
            embedder=row.embedder,
        )

    async def current_fingerprints(
        self, database: str
    ) -> dict[str, CurrentFingerprint]:
        """The current fingerprint of every document in `database`, by document id."""
        stmt = sa.select(
            fingerprints.c.id, fingerprints.c.document_id, fingerprints.c.change_key
        ).where(
            fingerprints.c.database == database, fingerprints.c.ended_sweep.is_(None)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return {
            row.document_id: CurrentFingerprint(
                id=row.id, document_id=row.document_id, change_key=row.change_key
            )
            for row in rows
        }

    async def history(self, database: str, document_id: str) -> list[Fingerprint]:
        """Every fingerprint of a document, oldest first."""
        stmt = (
            sa.select(fingerprints)
            .where(
                fingerprints.c.database == database,
                fingerprints.c.document_id == document_id,
            )
            .order_by(fingerprints.c.id)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [_fingerprint(row) for row in rows]

    async def chunk_text_hashes(self, fingerprint_id: int) -> list[str]:
        stmt = sa.select(chunk_texts.c.text_hash).where(
            chunk_texts.c.fingerprint_id == fingerprint_id
        )
        async with self._engine.connect() as conn:
            return list((await conn.execute(stmt)).scalars())

    async def record(self, sweep: DatabaseSweep) -> int:
        """Write one database's sweep and everything it found in one transaction; returns the sweep id."""
        async with self._engine.begin() as conn:
            sweep_id = await self._insert_sweep(conn, sweep)
            if sweep.status == SweepStatus.OK:
                await self._end(conn, sweep.replaced, sweep_id, deleted=False)
                await self._end(conn, sweep.deleted, sweep_id, deleted=True)
                await self._insert_fingerprints(conn, sweep, sweep_id)
                await self._refresh(conn, sweep)
        return sweep_id

    async def _insert_sweep(self, conn: AsyncConnection, sweep: DatabaseSweep) -> int:
        result = await conn.execute(
            sa.insert(sweeps).values(
                database=sweep.database,
                started_at=sweep.started_at,
                finished_at=sweep.finished_at,
                status=sweep.status,
                table_versions=(
                    json.dumps(sweep.table_versions)
                    if sweep.table_versions is not None
                    else None
                ),
                embedder=sweep.embedder,
                rebaseline=sweep.rebaseline,
                error=sweep.error,
                documents=sweep.documents,
                changed=len(sweep.new),
                deleted=len(sweep.deleted),
            )
        )
        (sweep_id,) = result.inserted_primary_key or (None,)
        assert sweep_id is not None
        return sweep_id

    async def _end(
        self, conn: AsyncConnection, ids: list[int], sweep_id: int, *, deleted: bool
    ) -> None:
        if not ids:
            return
        await conn.execute(
            sa.update(fingerprints)
            .where(fingerprints.c.id.in_(ids))
            .values(ended_sweep=sweep_id, deleted=deleted)
        )
        await conn.execute(
            sa.delete(chunk_texts).where(chunk_texts.c.fingerprint_id.in_(ids))
        )

    async def _insert_fingerprints(
        self, conn: AsyncConnection, sweep: DatabaseSweep, sweep_id: int
    ) -> None:
        for new in sweep.new:
            result = await conn.execute(
                sa.insert(fingerprints).values(
                    database=sweep.database,
                    document_id=new.document_id,
                    uri=new.uri,
                    title=new.title,
                    change_key=new.change_key,
                    md5=new.md5,
                    content_type=new.content_type,
                    source_revision=new.source_revision,
                    metadata_keys=json.dumps(new.metadata_keys),
                    chunks=new.chunks,
                    embedded_chunks=new.embedded_chunks,
                    chars=new.chars,
                    centroid=new.centroid,
                    embedder=new.embedder,
                    replacement_chars=new.replacement_chars,
                    chunk_stats=json.dumps(new.chunk_stats),
                    created_at=new.created_at,
                    updated_at=new.updated_at,
                    became_current_sweep=sweep_id,
                )
            )
            (fingerprint_id,) = result.inserted_primary_key or (None,)
            if new.chunk_texts:
                await conn.execute(
                    sa.insert(chunk_texts),
                    [
                        {
                            "database": sweep.database,
                            "fingerprint_id": fingerprint_id,
                            "text_hash": text_hash,
                            "chars": chars,
                        }
                        for text_hash, chars in new.chunk_texts
                    ],
                )

    async def _refresh(self, conn: AsyncConnection, sweep: DatabaseSweep) -> None:
        if not sweep.refreshed:
            return
        await conn.execute(
            sa.update(fingerprints)
            .where(fingerprints.c.id == sa.bindparam("fingerprint_id"))
            .values(
                uri=sa.bindparam("new_uri"),
                title=sa.bindparam("new_title"),
                source_revision=sa.bindparam("new_source_revision"),
                metadata_keys=sa.bindparam("new_metadata_keys"),
                updated_at=sa.bindparam("new_updated_at"),
            ),
            [
                {
                    "fingerprint_id": refresh.fingerprint_id,
                    "new_uri": refresh.uri,
                    "new_title": refresh.title,
                    "new_source_revision": refresh.source_revision,
                    "new_metadata_keys": json.dumps(refresh.metadata_keys),
                    "new_updated_at": refresh.updated_at,
                }
                for refresh in sweep.refreshed
            ],
        )


def _fingerprint(row: sa.Row) -> Fingerprint:
    return Fingerprint(
        id=row.id,
        database=row.database,
        document_id=row.document_id,
        uri=row.uri,
        title=row.title,
        change_key=row.change_key,
        md5=row.md5,
        source_revision=row.source_revision,
        metadata_keys=json.loads(row.metadata_keys),
        chunks=row.chunks,
        embedded_chunks=row.embedded_chunks,
        chars=row.chars,
        centroid=row.centroid,
        embedder=row.embedder,
        replacement_chars=row.replacement_chars,
        chunk_stats=json.loads(row.chunk_stats),
        became_current_sweep=row.became_current_sweep,
        ended_sweep=row.ended_sweep,
        deleted=row.deleted,
    )
