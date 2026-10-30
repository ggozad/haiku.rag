import sqlalchemy as sa
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.schema import CreateIndex, DropIndex

from haiku.rag.config.models import QueueConfig
from haiku.rag.ingester.exceptions import QueueError
from haiku.rag.ingester.queue.db import (
    SCHEMA_VERSION,
    install_sqlite_pragmas,
    jobs,
    metadata,
    schema_version,
    sync_state,
)

__all__ = [
    "SCHEMA_VERSION",
    "apply_migrations",
    "make_engine",
    "open_queue",
    "verify_schema",
]

# pg_advisory_xact_lock key shared by every haiku.rag queue migration.
_MIGRATION_LOCK = 0x6861696B7571


def make_engine(config: QueueConfig) -> AsyncEngine:
    """Build the queue's AsyncEngine from config. Uses `dburi` when set,
    otherwise a `sqlite+aiosqlite` URL pointing at the resolved `path`
    (creating the parent directory). SQLite runs in WAL mode with a small pool
    so reads (API stats/jobs) proceed concurrently with worker writes; the
    claim stays atomic via its single UPDATE statement, not the pool size.
    Postgres uses pool_pre_ping so a long-running ingester survives a DB
    restart or idle connection drop, and holds at most five connections per
    process."""
    if config.dburi:
        url = make_url(config.dburi)
    else:
        path = config.path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # URL.create keeps the path literal — building a string and reparsing
        # would treat `?`/`#` in the filename as query/fragment.
        url = URL.create("sqlite+aiosqlite", database=str(path))

    if url.get_backend_name() == "sqlite":
        engine = create_async_engine(url, pool_size=5, max_overflow=5)
    else:
        engine = create_async_engine(
            url, pool_pre_ping=True, pool_size=2, max_overflow=3
        )
    install_sqlite_pragmas(engine)
    return engine


def _target(engine: AsyncEngine) -> str:
    return engine.url.render_as_string(hide_password=True)


async def _current_version(conn: AsyncConnection) -> int | None:
    """The recorded schema version, None when the table is missing or empty."""
    present = await conn.run_sync(
        lambda sync: sa.inspect(sync).has_table("schema_version")
    )
    if not present:
        return None
    return (
        await conn.execute(sa.select(schema_version.c.version).limit(1))
    ).scalar_one_or_none()


def _newer_than_this_release(engine: AsyncEngine, current: int) -> QueueError:
    return QueueError(
        f"queue at {_target(engine)} is at schema {current}, newer than this "
        f"release's {SCHEMA_VERSION}: upgrade haiku.rag"
    )


async def verify_schema(engine: AsyncEngine) -> None:
    """Refuse a queue whose schema is missing, behind or ahead of this release.

    Writes nothing: an account with SELECT on `schema_version` can run it."""
    async with engine.connect() as conn:
        current = await _current_version(conn)
    if current is None:
        raise QueueError(
            f"queue at {_target(engine)} has no schema: run `haiku-ingester "
            "queue init --dburi URL` as the account that creates its tables"
        )
    if current < SCHEMA_VERSION:
        raise QueueError(
            f"queue at {_target(engine)} is at schema {current} and this release "
            f"needs {SCHEMA_VERSION}: run `haiku-ingester queue migrate --dburi "
            "URL` as the account that owns its tables"
        )
    if current > SCHEMA_VERSION:
        raise _newer_than_this_release(engine, current)


async def apply_migrations(engine: AsyncEngine) -> int:
    """Idempotently create tables/indexes and pin schema_version.

    Returns the schema version after the call. Safe on a fresh DB or one
    already at the latest version. Refuses a newer one before any DDL.
    """
    async with engine.begin() as conn:
        if conn.dialect.name == "sqlite":
            # sqlite3 opens a transaction before DML only. The DDL below rolls
            # back with the block only inside one opened here.
            await conn.exec_driver_sql("BEGIN")
        else:  # pragma: no cover - Postgres, run by the integration suite
            await conn.execute(
                sa.text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _MIGRATION_LOCK},
            )
        current = await _current_version(conn)
        if current is not None and current > SCHEMA_VERSION:
            raise _newer_than_this_release(engine, current)
        await conn.run_sync(metadata.create_all)
        if current is None:
            await conn.execute(sa.insert(schema_version).values(version=SCHEMA_VERSION))
        elif current < SCHEMA_VERSION:
            # create_all only creates missing tables/indexes, never adds columns
            # to an existing table, so column additions need explicit ALTERs.
            if current < 2:
                await conn.execute(
                    sa.text("ALTER TABLE jobs ADD COLUMN last_heartbeat_at TEXT")
                )
                await conn.execute(
                    sa.text(
                        "UPDATE jobs SET last_heartbeat_at = claimed_at "
                        "WHERE status = 'claimed'"
                    )
                )
            if current < 3:
                await conn.execute(
                    sa.text(
                        "ALTER TABLE jobs ADD COLUMN conversion_stalled "
                        "BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )
            if current < 4:
                await conn.execute(
                    sa.text(
                        "ALTER TABLE jobs ADD COLUMN namespace TEXT NOT NULL DEFAULT ''"
                    )
                )
                # Every job index leads with the column just added. Emitted
                # from the Index objects create_all uses, so a migrated queue
                # enforces exactly what a new one does.
                for index in jobs.indexes:
                    await conn.execute(DropIndex(index, if_exists=True))
                    await conn.execute(CreateIndex(index))
                if conn.dialect.name == "sqlite":
                    # SQLite cannot alter a primary key in place.
                    await conn.execute(
                        sa.text("ALTER TABLE sync_state RENAME TO sync_state_v3")
                    )
                    await conn.run_sync(sync_state.create)
                    await conn.execute(
                        sa.text(
                            "INSERT INTO sync_state (namespace, source_id, uri, "
                            "revision, content_hash, last_seen_at, "
                            "last_ingested_at) SELECT '', source_id, uri, "
                            "revision, content_hash, last_seen_at, "
                            "last_ingested_at FROM sync_state_v3"
                        )
                    )
                    await conn.execute(sa.text("DROP TABLE sync_state_v3"))
                else:  # pragma: no cover - Postgres DDL, run by the integration suite
                    await conn.execute(
                        sa.text(
                            "ALTER TABLE sync_state ADD COLUMN namespace "
                            "TEXT NOT NULL DEFAULT ''"
                        )
                    )
                    await conn.execute(
                        sa.text(
                            "ALTER TABLE sync_state "
                            "DROP CONSTRAINT sync_state_pkey, "
                            "ADD PRIMARY KEY (namespace, source_id, uri)"
                        )
                    )
            if current < 6:  # pragma: no branch - 6 is SCHEMA_VERSION
                (index,) = (
                    i for i in jobs.indexes if i.name == "idx_jobs_source_claimable"
                )
                await conn.execute(CreateIndex(index, if_not_exists=True))
            await conn.execute(sa.update(schema_version).values(version=SCHEMA_VERSION))
    return SCHEMA_VERSION


async def open_queue(config: QueueConfig) -> AsyncEngine:
    """The queue engine with its schema checked: a SQLite file is this
    process's and migrates here, a server queue is migrated by the `queue`
    commands as the account that owns it and is only verified."""
    engine = make_engine(config)
    try:
        if engine.dialect.name == "sqlite":
            await apply_migrations(engine)
        else:  # pragma: no cover - Postgres, run by the integration suite
            await verify_schema(engine)
    except BaseException:
        await engine.dispose()
        raise
    return engine
