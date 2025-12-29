import json

import pytest

from haiku.rag.app import HaikuRAGApp


@pytest.mark.asyncio
async def test_app_info_outputs(temp_db_path, capsys):
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

    # Insert one of each - using the new config format
    settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": "1.2.3",
                        "embeddings": {
                            "model": {
                                "provider": "openai",
                                "name": "text-embedding-3-small",
                                "vector_dim": 3,
                            }
                        },
                    }
                ),
            )
        ]
    )
    docs_tbl.add([DocumentRecord(id="doc-1", content="hello")])
    chunks_tbl.add(
        [ChunkRecord(id="c1", document_id="doc-1", content="c", vector=[0.1, 0.2, 0.3])]
    )

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out
    # Validate expected content substrings
    # Note: Rich console may wrap long paths to new lines, so check separately
    assert "path:" in out
    assert str(temp_db_path) in out
    assert "haiku.rag version (db):" in out
    assert "embeddings: openai/text-embedding-3-small (dim: 3)" in out
    assert "documents: 1" in out
    assert "chunks: 1" in out

    # Vector index should not exist (only 1 chunk, need 256)
    assert "vector index: ✗ not created" in out
    assert "need 255 more chunks" in out

    # Table versions should be shown
    assert "versions (documents):" in out
    assert "versions (chunks):" in out

    # Package versions section
    assert "lancedb:" in out
    assert "haiku.rag:" in out


@pytest.mark.asyncio
async def test_app_info_with_vector_index(temp_db_path, capsys):
    # Build a database with enough chunks to create a vector index
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

    # Insert settings
    settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings='{"version": "1.0.0", "embeddings": {"model": {"provider": "ollama", "name": "test", "vector_dim": 3}}}',
            )
        ]
    )

    # Insert document
    docs_tbl.add([DocumentRecord(id="doc-1", content="test")])

    # Insert 512 chunks to allow index creation (PQ needs more than 256 for training)
    chunks = [
        ChunkRecord(
            id=f"chunk-{i}",
            document_id="doc-1",
            content=f"content {i}",
            vector=[0.1 * i, 0.2 * i, 0.3 * i],
        )
        for i in range(512)
    ]
    chunks_tbl.add(chunks)

    # Create vector index
    chunks_tbl.create_index(metric="cosine", index_type="IVF_PQ")

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out

    # Check vector index exists
    assert "vector index: ✓ exists" in out
    assert "indexed chunks: 512" in out
    assert "unindexed chunks: 0" in out

    # Check basic info still present
    assert "documents: 1" in out
    assert "chunks: 512" in out
