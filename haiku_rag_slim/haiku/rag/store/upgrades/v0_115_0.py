import json

import pyarrow as pa

from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import chunk_media
from haiku.rag.store.schema import ensure_indexes
from haiku.rag.store.upgrades import Upgrade

_BATCH_ROWS = 10_000
_COLUMNS = pa.schema(
    [
        pa.field("media", pa.string()),
        pa.field("start_time", pa.float64()),
        pa.field("end_time", pa.float64()),
    ]
)


async def _apply_item_times(store: Store) -> None:
    """Add the nullable `start_time` and `end_time` columns to `document_items`."""
    existing = {f.name for f in await store.document_items_table.schema()}
    missing = [
        pa.field(name, pa.float64())
        for name in ("start_time", "end_time")
        if name not in existing
    ]
    if missing:
        await store.document_items_table.add_columns(pa.schema(missing))


upgrade_item_times = Upgrade(
    version="0.115.0",
    apply=_apply_item_times,
    description="Add start_time and end_time to document items",
)


async def _apply_chunk_media(store: Store) -> None:
    """Add `media`, `start_time` and `end_time` to `chunks`, filled from each
    recording chunk's metadata. A rerun finishes an interrupted fill."""
    table = store.chunks_table
    existing = {f.name for f in await table.schema()}
    missing = [f for f in _COLUMNS if f.name not in existing]
    if missing:
        await table.add_columns(pa.schema(missing))

    stream = (
        await table.query()
        .where("media IS NULL AND metadata LIKE '%\"time_range\"%'")
        .select(["id", "metadata"])
        .to_batches(max_batch_length=_BATCH_ROWS)
    )
    async for batch in stream:
        rows = []
        for row in batch.to_pylist():
            media, start_time, end_time = chunk_media(json.loads(row["metadata"]))
            if media is not None:
                rows.append(
                    {
                        "id": row["id"],
                        "media": media,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
        if rows:
            source = pa.Table.from_pylist(
                rows, schema=pa.schema([pa.field("id", pa.string()), *_COLUMNS])
            )
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .use_index(False)
                .execute(source)
            )
    await ensure_indexes(table, "chunks", store._config.search.fts_language)


upgrade_chunk_media = Upgrade(
    version="0.115.0",
    apply=_apply_chunk_media,
    description="Add media, start_time and end_time to chunks",
)
