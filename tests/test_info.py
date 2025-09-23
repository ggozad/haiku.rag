import json

import pytest

from haiku.rag.app import HaikuRAGApp


@pytest.mark.asyncio
async def test_app_info_outputs_and_read_only(temp_db_path, capsys):
    # Build a minimal LanceDB with settings, documents, and chunks without using Store
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    db = lancedb.connect(temp_db_path)

    class SettingsRecord(LanceModel):
        id: str = Field(default="settings")
        settings: str = Field(default="{}")

    class DocumentRecord(LanceModel):
        id: str
        content: str

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        vector: Vector(3)  # type: ignore

    settings_tbl = db.create_table("settings", schema=SettingsRecord)
    docs_tbl = db.create_table("documents", schema=DocumentRecord)
    chunks_tbl = db.create_table("chunks", schema=ChunkRecord)

    # Insert one of each
    settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": "1.2.3",
                        "EMBEDDINGS_PROVIDER": "openai",
                        "EMBEDDINGS_MODEL": "text-embedding-3-small",
                        "EMBEDDINGS_VECTOR_DIM": 3,
                    }
                ),
            )
        ]
    )
    docs_tbl.add([DocumentRecord(id="doc-1", content="hello")])
    chunks_tbl.add(
        [ChunkRecord(id="c1", document_id="doc-1", content="c", vector=[0.1, 0.2, 0.3])]
    )

    # Capture versions before
    before_versions = {
        "settings": int(settings_tbl.version),
        "documents": int(docs_tbl.version),
        "chunks": int(chunks_tbl.version),
    }

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out
    # Validate expected content substrings
    assert f"path: \n{temp_db_path}" in out
    assert "haiku.rag version (db): 1.2.3" in out
    assert "embeddings: openai/text-embedding-3-small (dim: 3)" in out
    assert "lancedb:" in out
    assert "documents: 1" in out

    # Verify no versions changed (read-only)
    # Re-open to ensure fresh view
    db2 = lancedb.connect(temp_db_path)
    assert int(db2.open_table("settings").version) == before_versions["settings"]
    assert int(db2.open_table("documents").version) == before_versions["documents"]
    assert int(db2.open_table("chunks").version) == before_versions["chunks"]
