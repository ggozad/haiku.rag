import json

from lancedb.index import FTS
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade


async def _apply_add_content_fts(store: Store) -> None:  # pragma: no cover
    """Add content_fts column with contextualized content for better FTS."""
    # Read existing chunks
    try:
        chunks_arrow = await store.chunks_table.query().to_arrow()
        rows = chunks_arrow.to_pylist()
    except Exception:
        return

    if not rows:
        return

    # Infer vector dimensions from first row
    vec = rows[0].get("vector")
    if not isinstance(vec, list) or not vec:
        return
    vector_dim = len(vec)

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        content_fts: str = Field(default="")
        metadata: str = Field(default="{}")
        order: int = Field(default=0)
        vector: Vector(vector_dim) = Field(  # type: ignore
            default_factory=lambda: [0.0] * vector_dim
        )

    # Drop and recreate table with new schema
    try:
        await store.db.drop_table("chunks")
    except Exception:
        pass

    store.chunks_table = await store.db.create_table("chunks", schema=ChunkRecord)

    # Populate content_fts with contextualized content
    new_records: list[ChunkRecord] = []
    for row in rows:
        metadata_raw = row.get("metadata") or "{}"
        try:
            metadata = (
                json.loads(metadata_raw)
                if isinstance(metadata_raw, str)
                else metadata_raw
            )
        except Exception:
            metadata = {}

        headings = metadata.get("headings") if isinstance(metadata, dict) else None
        content = row.get("content", "")

        # Build contextualized content for FTS
        if headings:
            content_fts = "\n".join(headings) + "\n" + content
        else:
            content_fts = content

        new_records.append(
            ChunkRecord(
                id=row.get("id"),
                document_id=row.get("document_id"),
                content=content,
                content_fts=content_fts,
                metadata=metadata_raw,
                order=row.get("order", 0),
                vector=row.get("vector") or [0.0] * vector_dim,
            )
        )

    if new_records:
        await store.chunks_table.add(new_records)

    # Drop old FTS index on content column if it exists
    try:
        await store.chunks_table.drop_index("content_idx")
    except Exception:
        pass

    # Create FTS index on content_fts
    await store.chunks_table.create_index(
        "content_fts",
        config=FTS(with_position=True, remove_stop_words=False),
        replace=True,
    )


upgrade_contextualize_chunks = Upgrade(
    version="0.23.1",
    apply=_apply_add_content_fts,
    description="Add content_fts column for contextualized FTS search",
)
