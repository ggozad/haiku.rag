from pathlib import Path

import aiosqlite

from haiku.rag.ingester.queue.schema import ALL_DDL, SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "apply_migrations", "open_queue"]


async def _exec(conn: aiosqlite.Connection, sql: str, *params) -> None:
    """Execute a statement and finalize the cursor. aiosqlite cursors stay
    attached until closed, blocking subsequent commits with 'SQL statements
    in progress'."""
    async with conn.execute(sql, params):
        pass


async def apply_migrations(conn: aiosqlite.Connection) -> int:
    """Idempotently create tables/indexes/views and pin schema_version.

    Returns the schema version after the call. Safe to call on a fresh DB
    or on one already at the latest version.
    """
    await _exec(conn, "PRAGMA journal_mode=WAL")
    await _exec(conn, "PRAGMA synchronous=NORMAL")
    await _exec(conn, "PRAGMA foreign_keys=ON")

    for stmt in ALL_DDL:
        await _exec(conn, stmt)

    async with conn.execute("SELECT version FROM schema_version LIMIT 1") as cursor:
        row = await cursor.fetchone()
    if row is None:
        await _exec(
            conn, "INSERT INTO schema_version (version) VALUES (?)", SCHEMA_VERSION
        )
    else:
        current = row[0]
        if current < SCHEMA_VERSION:  # pragma: no cover - no migrations yet
            # No diff migrations exist yet — future versions add UPDATE/ALTER
            # statements between here and the version bump.
            await _exec(conn, "UPDATE schema_version SET version = ?", SCHEMA_VERSION)

    await conn.commit()
    return SCHEMA_VERSION


async def open_queue(path: str | Path) -> aiosqlite.Connection:
    """Open the queue database at `path`, creating it (and the parent dir)
    if needed, and ensuring the schema is up-to-date."""
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await apply_migrations(conn)
    return conn
