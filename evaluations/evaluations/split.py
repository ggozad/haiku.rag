from collections.abc import Sequence
from pathlib import Path

from evaluations.datasets.collection_routing import shard_of
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.engine import Store
from haiku.rag.store.schema import ensure_indexes
from haiku.rag.utils.sql import escape_sql_string

DOCUMENT_COLUMN = {
    "documents": "id",
    "document_meta": "id",
    "chunks": "document_id",
    "document_items": "document_id",
}
"""The tables a document's rows live in, and the column naming the document."""


async def split_database(
    source: Path,
    destinations: Sequence[Path],
    config: AppConfig,
    *,
    batch_size: int = 32,
) -> list[int]:
    """Copy every document of `source` into the destination its uri hashes to.

    Rows are copied verbatim, ids, chunk vectors, item picture bytes and docling
    blobs included, so a shard holds exactly what the source held for its
    documents. Each destination records the embedder `config` names, which must
    be the one the source was built with. Returns the document count per
    destination.
    """
    async with HaikuRAG(
        source, config=config, read_only=True, skip_validation=True
    ) as src:
        shards: list[list[str]] = [[] for _ in destinations]
        for document in await src.list_documents():
            if document.uri is None:
                raise ValueError(f"document {document.id} has no uri to shard by")
            assert document.id is not None
            shards[shard_of(document.uri, len(destinations))].append(document.id)
        for ids, destination in zip(shards, destinations, strict=True):
            async with HaikuRAG(destination, config=config, create=True) as dst:
                for start in range(0, len(ids), batch_size):
                    await _copy_rows(
                        src.store, dst.store, ids[start : start + batch_size]
                    )
                for name in DOCUMENT_COLUMN:
                    await ensure_indexes(_table(dst.store, name), name)
    return [len(ids) for ids in shards]


def _table(store: Store, name: str):
    return getattr(store, f"{name}_table")


async def _copy_rows(src: Store, dst: Store, ids: list[str]) -> None:
    listed = ", ".join(f"'{escape_sql_string(i)}'" for i in ids)
    async with dst.write_transaction():
        for name, column in DOCUMENT_COLUMN.items():
            rows = (
                await _table(src, name)
                .query()
                .where(f"{column} IN ({listed})")
                .to_arrow()
            )
            if rows.num_rows:
                await _table(dst, name).add(rows)
