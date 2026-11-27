import asyncio
import hashlib
import logging
from collections.abc import Iterator

import lance
import pyarrow as pa

from haiku.rag.store.engine import (
    COMPACTION_TARGET_KEY,
    ConnectionMode,
    Store,
    kept_target,
    size_compaction,
)
from haiku.rag.store.schema import (
    ensure_indexes,
    get_documents_arrow_schema,
    is_blob,
)
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)

# Rows per read while rewriting. An original can run to hundreds of MB.
_BATCH_ROWS = 8


async def _apply_blob_originals(store: Store) -> None:
    """Rewrite `documents` with `original` as a lance blob column.

    Each part runs again on a rerun, so a migration stopped after the rewrite
    completes. Vacuums at zero retention, so no other process may use the
    database while it runs.
    """
    if not is_blob((await store.documents_table.schema()).field("original")):
        if store._connection_mode == ConnectionMode.CLOUD:
            await _rewrite_through_lancedb(store)
        else:
            await _rewrite_through_lance(store)
    await ensure_indexes(
        store.documents_table, "documents", store._config.search.fts_language
    )

    vacuumed = await store.vacuum(retention_seconds=0)
    if not vacuumed and store._connection_mode != ConnectionMode.CLOUD:
        raise RuntimeError(
            "The vacuum was skipped, so the documents table is stored twice. "
            "Run 'haiku-rag migrate' again."
        )

    tags = sorted(await store.documents_table.tags.list())
    if tags:
        logger.warning(
            "Tags taken before this migration keep the documents table as it "
            f"was: {', '.join(tags)}. Delete them and run 'haiku-rag vacuum "
            "--retention-seconds 0' to free the space."
        )


async def _rewrite_through_lance(store: Store) -> None:
    """One overwrite at lance format 2.2, in fragments of the compaction target.

    The overwrite keeps the table's history, tags and config, and drops its
    indexes. The target is stored first, so a vacuum keeps the fragments
    written here. A table with an oversized or unmeasurable fragment is written
    one row per fragment, and vacuum merges the rows that fit.
    """
    table = store.documents_table
    dataset = await table.to_lance()
    sizing = await asyncio.to_thread(
        size_compaction, dataset, store._config.storage.compaction_target_bytes
    )
    rows = None
    if sizing is not None and not sizing.oversized:
        stored = dataset.config().get(COMPACTION_TARGET_KEY)
        rows = kept_target(int(stored) if stored else None, sizing.target_rows)
    if rows is not None:
        await store._run_lance_maintenance(
            table, lambda ds: ds.update_config({COMPACTION_TARGET_KEY: str(rows)})
        )
    schema = get_documents_arrow_schema()
    await store._run_lance_maintenance(
        table,
        lambda ds: lance.write_dataset(
            pa.RecordBatchReader.from_batches(schema, _as_blobs(ds, schema)),
            ds.uri,
            schema=schema,
            mode="overwrite",
            data_storage_version="2.2",
            max_rows_per_file=rows or 1,
            storage_options=store._config.lancedb.storage_options or None,
        ),
    )


def _as_blobs(
    dataset: lance.LanceDataset, schema: pa.Schema
) -> Iterator[pa.RecordBatch]:
    """The dataset's rows with each original as a blob value and its SHA-256,
    and null in the columns of `schema` the dataset lacks.

    lance stores a `large_binary` source written under a blob schema as
    `large_binary`, so the values are built as blobs here.
    """
    for batch in dataset.to_batches(
        columns=[name for name in schema.names if name in dataset.schema.names],
        batch_size=_BATCH_ROWS,
        batch_readahead=1,
        fragment_readahead=1,
    ):
        rows = batch.to_pylist()
        for row in rows:
            original = row["original"]
            row["original_sha256"] = _sha256(original)
            if original is not None:
                row["original"] = {
                    "data": original,
                    "uri": None,
                    "position": None,
                    "size": None,
                }
        yield pa.RecordBatch.from_pylist(rows, schema=schema)


def _sha256(original: bytes | None) -> str | None:
    return None if original is None else hashlib.sha256(original).hexdigest()


async def _rewrite_through_lancedb(store: Store) -> None:
    """The rewrite on LanceDB Cloud, where pylance cannot reach the table."""
    # ponytail: reads the whole table into memory; stream it if a Cloud
    # database outgrows that.
    schema = get_documents_arrow_schema()
    rows = await store.documents_table.query().to_arrow()
    hashes = pa.array(
        [_sha256(o) for o in rows["original"].to_pylist()], type=pa.string()
    )
    rows = rows.append_column("original_sha256", hashes)
    for field in schema:
        if field.name not in rows.schema.names:
            rows = rows.append_column(field, pa.nulls(len(rows), field.type))
    rows = rows.select(schema.names)
    store.documents_table = await store.db.create_table(
        "documents",
        rows,
        schema=schema,
        mode="overwrite",
        storage_options={"new_table_data_storage_version": "2.2"},
    )


upgrade_blob_originals = Upgrade(
    version="0.114.0",
    apply=_apply_blob_originals,
    description="Store originals as lance blobs",
)
