import json
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from haiku.rag.config.models import RepeatedChunksConfig
from haiku.rag.curate.store.db import (
    chunk_texts,
    fingerprints,
    flags,
    layout,
    sweeps,
    watched,
)
from haiku.rag.curate.store.models import (
    Change,
    ChangeKind,
    CurrentDocument,
    CurrentFingerprint,
    DatabaseSummary,
    DatabaseSweep,
    DatabaseView,
    Detection,
    Fingerprint,
    Flag,
    FlagKind,
    FlagStatus,
    LastSweep,
    RepeatedText,
    Revision,
    SweepStatus,
    Watch,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CurateRepository:
    """Reads and writes of curate's store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["StoreWriter"]:
        async with self._engine.begin() as conn:
            yield StoreWriter(conn)

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

    async def flags(
        self,
        *,
        database: str | None = None,
        kind: FlagKind | None = None,
        status: FlagStatus | None = None,
    ) -> list[Flag]:
        """Flags, oldest first, narrowed by any of the arguments given.

        `database` includes cross-database flags with a member in it.
        """
        stmt = sa.select(flags).order_by(flags.c.id)
        if database is not None:
            stmt = stmt.where(
                sa.or_(flags.c.database == database, flags.c.database.is_(None))
            )
        if kind is not None:
            stmt = stmt.where(flags.c.kind == kind)
        if status is not None:
            stmt = stmt.where(flags.c.status == status)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        found = [_flag(row) for row in rows]
        if database is None:
            return found
        return [
            flag
            for flag in found
            if flag.database is not None or _has_member(flag, database)
        ]

    async def flag(self, flag_id: int) -> Flag | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(sa.select(flags).where(flags.c.id == flag_id))
            ).first()
        return _flag(row) if row is not None else None

    async def databases(self, names: Iterable[str]) -> list[DatabaseSummary]:
        """One summary per database name, in the order given."""
        summaries = []
        across = await self.flags(status=FlagStatus.OPEN)
        across = [flag for flag in across if flag.database is None]
        async with self._engine.connect() as conn:
            for name in names:
                last = (
                    await conn.execute(
                        sa.select(sweeps)
                        .where(sweeps.c.database == name)
                        .order_by(sweeps.c.id.desc())
                        .limit(1)
                    )
                ).first()
                documents = (
                    await conn.execute(
                        sa.select(sa.func.count()).where(
                            fingerprints.c.database == name,
                            fingerprints.c.ended_sweep.is_(None),
                        )
                    )
                ).scalar_one()
                open_flags = (
                    await conn.execute(
                        sa.select(sa.func.count()).where(
                            flags.c.database == name,
                            flags.c.status == FlagStatus.OPEN,
                        )
                    )
                ).scalar_one()
                summaries.append(
                    DatabaseSummary(
                        database=name,
                        documents=documents,
                        open_flags=open_flags
                        + sum(_has_member(flag, name) for flag in across),
                        last_status=SweepStatus(last.status) if last else None,
                        last_sweep_at=last.finished_at if last else None,
                        last_error=last.error if last else None,
                        embedder=last.embedder if last else None,
                    )
                )
        return summaries

    async def changes(
        self, since: datetime | None, databases: Iterable[str]
    ) -> list[Change]:
        """Documents added, updated or deleted at or after `since` (all when None), oldest first."""
        names = list(databases)
        async with self._engine.connect() as conn:
            sweep_times = {
                row.id: row.started_at
                for row in await conn.execute(
                    sa.select(sweeps.c.id, sweeps.c.started_at).where(
                        sweeps.c.database.in_(names)
                    )
                )
            }
            rows = (
                await conn.execute(
                    sa.select(
                        fingerprints.c.id,
                        fingerprints.c.database,
                        fingerprints.c.document_id,
                        fingerprints.c.uri,
                        fingerprints.c.title,
                        fingerprints.c.became_current_sweep,
                        fingerprints.c.ended_sweep,
                        fingerprints.c.deleted,
                    )
                    .where(fingerprints.c.database.in_(names))
                    .order_by(fingerprints.c.id)
                )
            ).all()

        def included(at: str) -> bool:
            return since is None or datetime.fromisoformat(at) >= since

        seen: set[tuple[str, str]] = set()
        changes = []
        for row in rows:
            subject = (row.database, row.uri or row.document_id)
            became = sweep_times[row.became_current_sweep]
            if included(became):
                kind = ChangeKind.UPDATED if subject in seen else ChangeKind.ADDED
                changes.append(_change(kind, row, became))
            seen.add(subject)
            if row.deleted:
                ended = sweep_times[row.ended_sweep]
                if included(ended):
                    changes.append(_change(ChangeKind.DELETED, row, ended))
        return sorted(changes, key=lambda change: (change.at, change.fingerprint_id))

    async def documents(self, database: str) -> list[CurrentDocument]:
        """Current documents of `database` with their scores and open flags."""
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa.select(
                        fingerprints.c.id,
                        fingerprints.c.document_id,
                        fingerprints.c.uri,
                        fingerprints.c.title,
                        fingerprints.c.chunks,
                        fingerprints.c.chars,
                        fingerprints.c.replacement_chars,
                        fingerprints.c.chunk_stats,
                        layout.c.isolation,
                    )
                    .select_from(
                        fingerprints.outerjoin(
                            layout,
                            sa.and_(
                                layout.c.database == fingerprints.c.database,
                                layout.c.document_id == fingerprints.c.document_id,
                            ),
                        )
                    )
                    .where(
                        fingerprints.c.database == database,
                        fingerprints.c.ended_sweep.is_(None),
                    )
                    .order_by(fingerprints.c.id)
                )
            ).all()
            open_rows = (
                await conn.execute(
                    sa.select(
                        flags.c.kind, flags.c.fingerprint_id, flags.c.members
                    ).where(
                        flags.c.status == FlagStatus.OPEN,
                        sa.or_(
                            flags.c.database == database, flags.c.database.is_(None)
                        ),
                    )
                )
            ).all()
        by_fingerprint: dict[int, set[FlagKind]] = {}
        by_document: dict[str, set[FlagKind]] = {}
        for row in open_rows:
            kind = FlagKind(row.kind)
            if row.fingerprint_id is not None:
                by_fingerprint.setdefault(row.fingerprint_id, set()).add(kind)
            for member in json.loads(row.members) if row.members else []:
                if member["database"] == database:
                    by_document.setdefault(member["document_id"], set()).add(kind)
        return [
            CurrentDocument(
                database=database,
                document_id=row.document_id,
                uri=row.uri,
                title=row.title,
                fingerprint_id=row.id,
                chunks=row.chunks,
                chars=row.chars,
                replacement_chars=row.replacement_chars,
                chunk_stats=json.loads(row.chunk_stats),
                isolation=row.isolation,
                open_flags=sorted(
                    by_fingerprint.get(row.id, set())
                    | by_document.get(row.document_id, set())
                ),
            )
            for row in rows
        ]

    async def watches(self) -> list[Watch]:
        stmt = sa.select(watched).order_by(watched.c.database, watched.c.uri)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [
            Watch(database=r.database, uri=r.uri, note=r.note, added_at=r.added_at)
            for r in rows
        ]

    async def isolation(self, database: str) -> dict[str, float | None]:
        stmt = sa.select(layout.c.document_id, layout.c.isolation).where(
            layout.c.database == database
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return {row.document_id: row.isolation for row in rows}

    async def acknowledge(self, flag_id: int, note: str | None = None) -> bool:
        """Acknowledge a flag; False when there is no such flag."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                sa.update(flags)
                .where(flags.c.id == flag_id)
                .values(
                    status=FlagStatus.ACKNOWLEDGED, status_changed_at=_now(), note=note
                )
            )
        return result.rowcount > 0

    async def reopen(self, flag_id: int) -> bool:
        """Set an acknowledged flag back to open, dropping its note; False otherwise."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                sa.update(flags)
                .where(flags.c.id == flag_id, flags.c.status == FlagStatus.ACKNOWLEDGED)
                .values(status=FlagStatus.OPEN, status_changed_at=_now(), note=None)
            )
        return result.rowcount > 0

    async def watch(self, database: str, uri: str, note: str | None = None) -> None:
        """Watch `uri`; watching it again updates the note and keeps `added_at`."""
        async with self._engine.begin() as conn:
            updated = await conn.execute(
                sa.update(watched)
                .where(watched.c.database == database, watched.c.uri == uri)
                .values(note=note)
            )
            if updated.rowcount == 0:
                await conn.execute(
                    sa.insert(watched).values(
                        database=database, uri=uri, note=note, added_at=_now()
                    )
                )

    async def unwatch(self, database: str, uri: str) -> bool:
        """Stop watching `uri`; False when it was not watched."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                sa.delete(watched).where(
                    watched.c.database == database, watched.c.uri == uri
                )
            )
        return result.rowcount > 0


class StoreWriter:
    """Writes and reads inside one store transaction."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def record(self, sweep: DatabaseSweep) -> int:
        """Write one database's sweep and everything it found; returns the sweep id."""
        sweep_id = await self._insert_sweep(sweep)
        if sweep.status == SweepStatus.OK:
            await self._end(sweep.replaced, sweep_id, deleted=False)
            await self._end(sweep.deleted, sweep_id, deleted=True)
            await self._insert_fingerprints(sweep, sweep_id)
            await self._refresh(sweep)
        return sweep_id

    async def view(
        self, database: str, repeated_chunks: RepeatedChunksConfig
    ) -> DatabaseView:
        """The detectors' view of `database`, as the store holds it now.

        A revision's previous is the latest earlier revision of its subject that
        is not under an open or superseded `bad_update` flag.
        """
        unsettled = set(
            (
                await self._conn.execute(
                    sa.select(flags.c.fingerprint_id).where(
                        flags.c.database == database,
                        flags.c.kind == FlagKind.BAD_UPDATE,
                        flags.c.status.in_([FlagStatus.OPEN, FlagStatus.SUPERSEDED]),
                    )
                )
            ).scalars()
        )
        sweep_times = {
            row.id: row.started_at
            for row in await self._conn.execute(
                sa.select(sweeps.c.id, sweeps.c.started_at).where(
                    sweeps.c.database == database
                )
            )
        }
        rows = (
            await self._conn.execute(
                sa.select(*_REVISION_COLUMNS)
                .where(fingerprints.c.database == database)
                .order_by(fingerprints.c.id)
            )
        ).all()

        previous_of: dict[int, sa.Row] = {}
        latest: dict[str, sa.Row] = {}
        baseline: dict[str, sa.Row] = {}
        for row in rows:
            subject = row.uri or row.document_id
            if subject in baseline:
                previous_of[row.id] = baseline[subject]
            latest[subject] = row
            if row.id not in unsettled:
                baseline[subject] = row
        current = [row for row in rows if row.ended_sweep is None]
        current_subjects = {row.uri or row.document_id for row in current}
        previous = {
            row.id: previous_of[row.id] for row in current if row.id in previous_of
        }
        deletions = [
            row
            for subject, row in latest.items()
            if row.deleted and subject not in current_subjects
        ]
        centroids = await self._centroids(
            [row.id for row in current] + [row.id for row in previous.values()]
        )

        def revision(row: sa.Row) -> Revision:
            return _revision(row, centroids.get(row.id), sweep_times)

        watch_rows = (
            await self._conn.execute(
                sa.select(watched.c.uri, watched.c.added_at).where(
                    watched.c.database == database
                )
            )
        ).all()
        return DatabaseView(
            database=database,
            current=[revision(row) for row in current],
            previous={key: revision(row) for key, row in previous.items()},
            deletions=[revision(row) for row in deletions],
            watched={row.uri: row.added_at for row in watch_rows},
            repeated=await self._repeated(database, repeated_chunks),
        )

    async def current_revisions(self, databases: Iterable[str]) -> list[Revision]:
        """Current fingerprints of `databases`, without centroids."""
        rows = (
            await self._conn.execute(
                sa.select(*_REVISION_COLUMNS).where(
                    fingerprints.c.database.in_(list(databases)),
                    fingerprints.c.ended_sweep.is_(None),
                )
            )
        ).all()
        return [_revision(row, None, {}) for row in rows]

    async def reconcile(
        self,
        detections: list[Detection],
        database: str | None,
        current: list[Revision],
    ) -> None:
        """Bring `database`'s flags (cross-database flags for None) in line with `detections`.

        A flag no longer detected is superseded when the document moved on to a
        newer revision, and resolved otherwise.
        """
        current_ids = {revision.id for revision in current}
        current_subjects = {revision.subject for revision in current}
        now = _now()
        scope = (
            flags.c.database.is_(None)
            if database is None
            else flags.c.database == database
        )
        existing = {
            row.identity: row
            for row in (await self._conn.execute(sa.select(flags).where(scope))).all()
        }
        detected = set()
        for detection in detections:
            identity = detection.identity
            detected.add(identity)
            values = {
                "members": (
                    json.dumps(detection.members)
                    if detection.members is not None
                    else None
                ),
                "reasons": json.dumps(detection.reasons),
            }
            row = existing.get(identity)
            if row is None:
                await self._conn.execute(
                    sa.insert(flags).values(
                        identity=identity,
                        kind=detection.kind,
                        database=detection.database,
                        subject=detection.subject,
                        fingerprint_id=detection.fingerprint_id,
                        previous_fingerprint_id=detection.previous_fingerprint_id,
                        status=FlagStatus.OPEN,
                        raised_at=now,
                        status_changed_at=now,
                        **values,
                    )
                )
                continue
            if row.status == FlagStatus.RESOLVED:
                values |= {"status": FlagStatus.OPEN, "status_changed_at": now}
            await self._conn.execute(
                sa.update(flags).where(flags.c.id == row.id).values(**values)
            )
        for identity, row in existing.items():
            if identity in detected or row.status != FlagStatus.OPEN:
                continue
            if row.kind == FlagKind.WATCHED_DELETION:
                superseded = row.subject in current_subjects
            else:
                superseded = (
                    row.fingerprint_id is not None
                    and row.fingerprint_id not in current_ids
                )
            await self._conn.execute(
                sa.update(flags)
                .where(flags.c.id == row.id)
                .values(
                    status=(
                        FlagStatus.SUPERSEDED if superseded else FlagStatus.RESOLVED
                    ),
                    status_changed_at=now,
                )
            )

    async def write_layout(
        self, database: str, sweep_id: int, isolation: dict[str, float | None]
    ) -> None:
        await self._conn.execute(sa.delete(layout).where(layout.c.database == database))
        if isolation:
            await self._conn.execute(
                sa.insert(layout),
                [
                    {
                        "database": database,
                        "document_id": document_id,
                        "sweep_id": sweep_id,
                        "isolation": score,
                    }
                    for document_id, score in isolation.items()
                ],
            )

    async def _centroids(self, ids: list[int]) -> dict[int, bytes | None]:
        found: dict[int, bytes | None] = {}
        for start in range(0, len(ids), 500):
            rows = await self._conn.execute(
                sa.select(fingerprints.c.id, fingerprints.c.centroid).where(
                    fingerprints.c.id.in_(ids[start : start + 500])
                )
            )
            found.update({row.id: row.centroid for row in rows})
        return found

    async def _repeated(
        self, database: str, config: RepeatedChunksConfig
    ) -> list[RepeatedText]:
        shared = (
            sa.select(chunk_texts.c.text_hash)
            .where(
                chunk_texts.c.database == database,
                chunk_texts.c.chars >= config.min_chars,
            )
            .group_by(chunk_texts.c.text_hash)
            .having(
                sa.func.count(sa.distinct(chunk_texts.c.fingerprint_id))
                >= config.min_documents
            )
        )
        rows = (
            await self._conn.execute(
                sa.select(
                    chunk_texts.c.text_hash,
                    chunk_texts.c.chars,
                    fingerprints.c.document_id,
                )
                .join(fingerprints, fingerprints.c.id == chunk_texts.c.fingerprint_id)
                .where(
                    chunk_texts.c.database == database,
                    chunk_texts.c.text_hash.in_(shared),
                )
                .order_by(chunk_texts.c.text_hash)
            )
        ).all()
        by_hash: dict[str, RepeatedText] = {}
        for row in rows:
            repeated = by_hash.setdefault(
                row.text_hash, RepeatedText(row.text_hash, row.chars, [])
            )
            if row.document_id not in repeated.document_ids:
                repeated.document_ids.append(row.document_id)
        return list(by_hash.values())

    async def _insert_sweep(self, sweep: DatabaseSweep) -> int:
        result = await self._conn.execute(
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

    async def _end(self, ids: list[int], sweep_id: int, *, deleted: bool) -> None:
        if not ids:
            return
        await self._conn.execute(
            sa.update(fingerprints)
            .where(fingerprints.c.id.in_(ids))
            .values(ended_sweep=sweep_id, deleted=deleted)
        )
        await self._conn.execute(
            sa.delete(chunk_texts).where(chunk_texts.c.fingerprint_id.in_(ids))
        )

    async def _insert_fingerprints(self, sweep: DatabaseSweep, sweep_id: int) -> None:
        for new in sweep.new:
            result = await self._conn.execute(
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
                await self._conn.execute(
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

    async def _refresh(self, sweep: DatabaseSweep) -> None:
        if not sweep.refreshed:
            return
        await self._conn.execute(
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


_REVISION_COLUMNS = (
    fingerprints.c.id,
    fingerprints.c.database,
    fingerprints.c.document_id,
    fingerprints.c.uri,
    fingerprints.c.md5,
    fingerprints.c.embedder,
    fingerprints.c.chars,
    fingerprints.c.chunks,
    fingerprints.c.embedded_chunks,
    fingerprints.c.replacement_chars,
    fingerprints.c.metadata_keys,
    fingerprints.c.became_current_sweep,
    fingerprints.c.ended_sweep,
    fingerprints.c.deleted,
)


def _revision(
    row: sa.Row, centroid: bytes | None, sweep_times: dict[int, str]
) -> Revision:
    return Revision(
        id=row.id,
        database=row.database,
        document_id=row.document_id,
        subject=row.uri or row.document_id,
        md5=row.md5,
        embedder=row.embedder,
        centroid=centroid,
        chars=row.chars,
        chunks=row.chunks,
        embedded_chunks=row.embedded_chunks,
        replacement_chars=row.replacement_chars,
        metadata_keys=json.loads(row.metadata_keys),
        became_current_at=sweep_times.get(row.became_current_sweep, ""),
        ended_at=sweep_times.get(row.ended_sweep) if row.ended_sweep else None,
        deleted=row.deleted,
    )


def _has_member(flag: Flag, database: str) -> bool:
    return any(member["database"] == database for member in flag.members or [])


def _change(kind: ChangeKind, row: sa.Row, at: str) -> Change:
    return Change(
        kind=kind,
        database=row.database,
        document_id=row.document_id,
        uri=row.uri,
        title=row.title,
        fingerprint_id=row.id,
        at=at,
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


def _flag(row: sa.Row) -> Flag:
    return Flag(
        id=row.id,
        identity=row.identity,
        kind=FlagKind(row.kind),
        database=row.database,
        subject=row.subject,
        fingerprint_id=row.fingerprint_id,
        previous_fingerprint_id=row.previous_fingerprint_id,
        members=json.loads(row.members) if row.members is not None else None,
        reasons=json.loads(row.reasons),
        status=FlagStatus(row.status),
        raised_at=row.raised_at,
        status_changed_at=row.status_changed_at,
        note=row.note,
    )
