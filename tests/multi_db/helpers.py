"""Databases to run the multi-database tests against."""

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.models import Chunk
from haiku.rag.utils import locate_database


def _config(tmp_path, names) -> AppConfig:
    return AppConfig(
        lancedb=LanceDBConfig(
            databases={n: str(tmp_path / f"{n}.lancedb") for n in names}
        )
    )


async def _seed(config, name, contents):
    """Precomputed embeddings and FTS queries keep the embedder out of the way:
    these tests are about fusion, not retrieval quality."""
    dim = get_config().embeddings.model.vector_dim
    async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
        for content in contents:
            doc = DoclingDocument(name=content)
            doc.add_text(label=DocItemLabel.TEXT, text=content)
            await rag.import_document(
                doc,
                [Chunk(content=content, embedding=[0.1] * dim, order=0)],
                uri=f"test://{name}/{content}",
            )


async def _restore_embedder(config, name, *, provider=None, model_name=None):
    """Rewrite what one database records about the embedder that wrote it,
    standing in for a database built elsewhere with another model."""
    import json

    import lancedb

    db_path = locate_database(config.lancedb.databases[name])
    assert not isinstance(db_path, str)
    db = await lancedb.connect_async(str(db_path.resolve()))
    table = await db.open_table("settings")
    rows = (
        await table.query().where("id = 'settings'").limit(1).to_arrow()
    ).to_pylist()
    stored = json.loads(rows[0]["settings"])
    model = stored["embeddings"]["model"]
    if provider is not None:
        model["provider"] = provider
    if model_name is not None:
        model["name"] = model_name
    await table.update({"settings": json.dumps(stored)}, where="id = 'settings'")


async def _seed_expandable(config, name, sentences):
    """One document whose chunk covers a single item, so expansion has
    neighbours to pull in and rebuilds the result."""
    dim = get_config().embeddings.model.vector_dim
    doc = DoclingDocument(name=name)
    for sentence in sentences:
        doc.add_text(label=DocItemLabel.TEXT, text=sentence)
    async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
        await rag.import_document(
            doc,
            [
                Chunk(
                    content=sentences[0],
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={"doc_item_refs": ["#/texts/0"]},
                )
            ],
            uri=f"test://{name}/expandable",
        )


class StubReranker:
    """Scores the union, reversing it so the ordering is unmistakably its own."""

    def __init__(self):
        self.seen: list[str] = []

    async def rerank(self, query, chunks, top_n):
        self.seen = [c.content for c in chunks]
        # Whatever the caller attached before handing them over.
        self.attached = {
            c.content.split()[0]: c._picture_data
            for c in chunks
            if getattr(c, "_picture_data", None)
        }
        return [(c, 1.0 - i) for i, c in enumerate(reversed(chunks))][:top_n]
