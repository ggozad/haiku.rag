from pathlib import Path

import aiosqlite

from haiku.rag.ingester.queue.schema import ALL_DDL, SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "apply_migrations", "open_queue"]


async def apply_migrations(conn: aiosqlite.Connection) -> int:
    """Idempotently create tables/indexes/views and pin schema_version.

    Returns the schema version after the call. Safe to call on a fresh DB
    or on one already at the latest version.
    """
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA foreign_keys=ON")

    for stmt in ALL_DDL:
        await conn.execute(stmt)

    cursor = await conn.execute("SELECT version FROM schema_version LIMIT 1")
    row = await cursor.fetchone()
    if row is None:
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
    else:
        current = row[0]
        if current < SCHEMA_VERSION:
            # No diff migrations exist yet — future versions add UPDATE/ALTER
            # statements between here and the version bump.
            await conn.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )

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
