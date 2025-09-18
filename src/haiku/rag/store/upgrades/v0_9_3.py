import json

from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade


def _infer_vector_dim(store: Store) -> int:
    """Infer vector dimension from existing data; fallback to embedder config."""
    try:
        arrow = store.chunks_table.search().limit(1).to_arrow()
        rows = arrow.to_pylist()
        if rows:
            vec = rows[0].get("vector")
            if isinstance(vec, list) and vec:
                return len(vec)
    except Exception:
        pass
    # Fallback to configured embedder vector dim
    return getattr(store.embedder, "_vector_dim", 1024)


def _apply_chunk_order(store: Store) -> None:
    """Add integer 'order' column to chunks and backfill from metadata."""

    vector_dim = _infer_vector_dim(store)

    class ChunkRecordV2(LanceModel):
        id: str
        document_id: str
        content: str
        metadata: str = Field(default="{}")
        order: int = Field(default=0)
        vector: Vector(vector_dim) = Field(  # type: ignore
            default_factory=lambda: [0.0] * vector_dim
        )

    # Read existing chunks
    try:
        chunks_arrow = store.chunks_table.search().to_arrow()
        rows = chunks_arrow.to_pylist()
    except Exception:
        rows = []

    new_chunk_records: list[ChunkRecordV2] = []
    for row in rows:
        md_raw = row.get("metadata") or "{}"
        try:
            md = json.loads(md_raw) if isinstance(md_raw, str) else md_raw
        except Exception:
            md = {}
        # Extract and normalize order
        order_val = 0
        try:
            if isinstance(md, dict) and "order" in md:
                order_val = int(md["order"])  # type: ignore[arg-type]
        except Exception:
            order_val = 0

        if isinstance(md, dict) and "order" in md:
            md = {k: v for k, v in md.items() if k != "order"}

        vec = row.get("vector") or [0.0] * vector_dim

        new_chunk_records.append(
            ChunkRecordV2(
                id=row.get("id"),
                document_id=row.get("document_id"),
                content=row.get("content", ""),
                metadata=json.dumps(md),
                order=order_val,
                vector=vec,
            )
        )

    # Recreate chunks table with new schema
    try:
        store.db.drop_table("chunks")
    except Exception:
        pass

    store.chunks_table = store.db.create_table("chunks", schema=ChunkRecordV2)
    store.chunks_table.create_fts_index("content", replace=True)

    if new_chunk_records:
        store.chunks_table.add(new_chunk_records)


upgrade_order = Upgrade(
    version="0.9.3",
    apply=_apply_chunk_order,
    description="Add 'order' column to chunks and backfill from metadata",
)


def _apply_fts_phrase_support(store: Store) -> None:
    """Recreate FTS index with phrase query support and no stop-word removal."""
    try:
        store.chunks_table.create_fts_index(
            "content", replace=True, with_position=True, remove_stop_words=False
        )
    except Exception:
        pass


upgrade_fts_phrase = Upgrade(
    version="0.9.3",
    apply=_apply_fts_phrase_support,
    description="Enable FTS phrase queries (with positions) and keep stop-words",
)
