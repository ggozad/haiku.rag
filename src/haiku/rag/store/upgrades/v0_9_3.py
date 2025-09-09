from __future__ import annotations

import json
from typing import TYPE_CHECKING

from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from . import Upgrade

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from haiku.rag.store.engine import Store


def _apply_chunk_order(store: Store) -> None:
    """Add integer 'order' column to chunks and backfill from metadata."""
    vector_dim = store.embedder._vector_dim

    # ============== Chunks: add 'order' column and backfill ==============
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
    # Recreate FTS index on content
    store.chunks_table.create_fts_index("content", replace=True)

    if new_chunk_records:
        store.chunks_table.add(new_chunk_records)


upgrade_order = Upgrade(
    version="0.9.3",
    apply=_apply_chunk_order,
    description="Add 'order' column to chunks and backfill from metadata",
)
