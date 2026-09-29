from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def make_engine(path: Path, dburi: str | None) -> AsyncEngine:
    """Build an AsyncEngine for `dburi`, or for a SQLite file at `path` when unset.

    SQLite runs in WAL mode with a small pool; Postgres uses pool_pre_ping.
    """
    if dburi:
        url = make_url(dburi)
    else:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # URL.create keeps the path literal — building a string and reparsing
        # would treat `?`/`#` in the filename as query/fragment.
        url = URL.create("sqlite+aiosqlite", database=str(path))

    if url.get_backend_name() == "sqlite":
        engine = create_async_engine(url, pool_size=5, max_overflow=5)
    else:
        engine = create_async_engine(url, pool_pre_ping=True)
    install_sqlite_pragmas(engine)
    return engine


def display_target(path: Path, dburi: str | None) -> str:
    """Where the store is, for display, with any password in `dburi` masked."""
    if dburi:
        return make_url(dburi).render_as_string(hide_password=True)
    return str(path)


def install_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Register a connect listener that sets the per-connection pragmas.

    SQLite-only: asyncpg rejects PRAGMA, so the listener is never attached for
    Postgres.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()
