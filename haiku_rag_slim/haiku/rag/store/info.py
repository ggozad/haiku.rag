"""Read-only database information and its result models.

Every function here only reads: it reports what a database contains and how it
is configured, and never writes a table or a version.
"""

import json
from pathlib import Path

import lancedb
from pydantic import BaseModel, Field

from haiku.rag.config import AppConfig
from haiku.rag.store.engine import connect_lancedb
from haiku.rag.store.schema import REQUIRED_TABLES


async def get_database_stats(db: lancedb.AsyncConnection) -> dict:
    """Collect stats for every haiku.rag table on the connection.

    Missing tables return ``{"exists": False}``. Present tables include
    ``num_rows``, ``total_bytes``, and ``num_versions``. The ``chunks``
    entry additionally reports vector index status and, when an index
    exists, ``num_indexed_rows`` and ``num_unindexed_rows``.
    """
    existing = set((await db.list_tables()).tables)
    stats: dict = {}
    tables: dict = {}

    for name in REQUIRED_TABLES:
        if name not in existing:
            stats[name] = {"exists": False}
            continue
        tbl = await db.open_table(name)
        tables[name] = tbl
        # lancedb's .stats() stub claims TableStatistics but returns a plain dict at runtime.
        tbl_stats: dict = await tbl.stats()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        stats[name] = {
            "exists": True,
            "num_rows": tbl_stats.get("num_rows", 0),
            "total_bytes": tbl_stats.get("total_bytes", 0),
            "num_versions": len(await tbl.list_versions()),
        }

    if stats["chunks"]["exists"]:
        chunks_tbl = tables["chunks"]
        indices = await chunks_tbl.list_indices()
        has_vector_index = any("vector" in str(idx).lower() for idx in indices)
        stats["chunks"]["has_vector_index"] = has_vector_index
        if has_vector_index:
            index_stats = await chunks_tbl.index_stats("vector_idx")
            if index_stats is not None:
                stats["chunks"]["num_indexed_rows"] = index_stats.num_indexed_rows
                stats["chunks"]["num_unindexed_rows"] = index_stats.num_unindexed_rows

    return stats


class EmbeddingsInfo(BaseModel):
    provider: str = "unknown"
    name: str = "unknown"
    vector_dim: int | None = None


class TableInfo(BaseModel):
    name: str
    exists: bool
    num_rows: int = 0
    total_bytes: int = 0
    num_versions: int = 0


class VectorIndexInfo(BaseModel):
    exists: bool = False
    indexed_rows: int = 0
    unindexed_rows: int = 0


class PendingMigration(BaseModel):
    version: str
    description: str = ""


class DatabaseInfo(BaseModel):
    """Structured snapshot of a haiku.rag database, shared by the `info` CLI
    command and the ingester control plane. Read-only; gathered without
    opening a Store."""

    path: str
    exists: bool
    stored_version: str = "unknown"
    embeddings: EmbeddingsInfo = Field(default_factory=EmbeddingsInfo)
    tables: list[TableInfo] = Field(default_factory=list)
    vector_index: VectorIndexInfo = Field(default_factory=VectorIndexInfo)
    pending_migrations: list[PendingMigration] = Field(default_factory=list)
    packages: dict[str, str] = Field(default_factory=dict)


async def gather_database_info(location: Path | str, config: AppConfig) -> DatabaseInfo:
    """Collect read-only database state without going through Store, so a
    database missing tables (e.g. pre-migration) still reports what it can."""
    from haiku.rag.store.upgrades import get_pending_upgrades
    from haiku.rag.utils import get_package_versions

    display_path = str(location)

    db = await connect_lancedb(location, config)
    stats = await get_database_stats(db)

    if not any(entry["exists"] for entry in stats.values()):
        return DatabaseInfo(path=display_path, exists=False)

    stored_version = "unknown"
    embeddings = EmbeddingsInfo()
    if stats["settings"]["exists"]:
        settings_tbl = await db.open_table("settings")
        rows = (
            await settings_tbl.query().where("id = 'settings'").limit(1).to_arrow()
        ).to_pylist()
        if rows:
            raw = rows[0].get("settings") or "{}"
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
            stored_version = str(data.get("version", "unknown"))
            model = data.get("embeddings", {}).get("model", {})
            embeddings = EmbeddingsInfo(
                provider=model.get("provider", "unknown"),
                name=model.get("name", "unknown"),
                vector_dim=model.get("vector_dim"),
            )

    tables = [
        TableInfo(
            name=name,
            exists=stats[name]["exists"],
            num_rows=stats[name].get("num_rows", 0),
            total_bytes=stats[name].get("total_bytes", 0),
            num_versions=stats[name].get("num_versions", 0),
        )
        for name in ("documents", "document_meta", "chunks", "document_items")
    ]

    vector_index = VectorIndexInfo()
    if stats["chunks"]["exists"] and stats["chunks"].get("has_vector_index"):
        vector_index = VectorIndexInfo(
            exists=True,
            indexed_rows=stats["chunks"].get("num_indexed_rows", 0),
            unindexed_rows=stats["chunks"].get("num_unindexed_rows", 0),
        )

    pending = (
        get_pending_upgrades(stored_version) if stored_version != "unknown" else []
    )

    return DatabaseInfo(
        path=display_path,
        exists=True,
        stored_version=stored_version,
        embeddings=embeddings,
        tables=tables,
        vector_index=vector_index,
        pending_migrations=[
            PendingMigration(version=step.version, description=step.description or "")
            for step in pending
        ],
        packages=get_package_versions(),
    )
