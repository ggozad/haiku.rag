import json

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.store.schema import DocumentItemRecord


class _SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


class _DocumentRecord(LanceModel):
    id: str
    content: str


class _ChunkRecord(LanceModel):
    id: str
    document_id: str
    content: str
    vector: Vector(3)  # type: ignore


def openai_settings(version: str, **model: str) -> dict:
    return {
        "version": version,
        "embeddings": {
            "model": {
                "provider": "openai",
                "name": "text-embedding-3-small",
                "vector_dim": 3,
                **model,
            }
        },
    }


async def seed_database(
    path,
    settings: dict,
    *,
    with_document: bool = True,
    chunks: int = 1,
    with_items: bool = True,
) -> lancedb.AsyncTable:
    """Write `settings`, one document and `chunks` chunks; return the chunks table."""
    db = await lancedb.connect_async(path)
    settings_tbl = await db.create_table("settings", schema=_SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=_DocumentRecord)
    chunks_tbl = await db.create_table("chunks", schema=_ChunkRecord)
    if with_items:
        await db.create_table("document_items", schema=DocumentItemRecord)

    await settings_tbl.add([_SettingsRecord(settings=json.dumps(settings))])
    if with_document:
        await docs_tbl.add([_DocumentRecord(id="doc-1", content="hello")])
    if chunks:
        await chunks_tbl.add(
            [
                _ChunkRecord(
                    id=f"c{i}",
                    document_id="doc-1",
                    content=f"content {i}",
                    vector=[0.1 * i, 0.2 * i, 0.3 * i],
                )
                for i in range(1, chunks + 1)
            ]
        )
    return chunks_tbl
