from io import BytesIO

import httpx
import pytest
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DoclingDocument, ImageRef, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image
from pydantic_ai import models

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.models import Chunk
from haiku.rag.store.models.document_item import DocumentItem
from haiku_rag_app.main import create_app

models.ALLOW_MODEL_REQUESTS = False


def image_bytes(fmt: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format=fmt)
    return buffer.getvalue()


def paged_document(text: str) -> DoclingDocument:
    """A one-page document whose only item has a bounding box to draw."""
    doc = DoclingDocument(name=text)
    doc.add_page(
        page_no=1,
        size=Size(width=612.0, height=792.0),
        image=ImageRef.from_pil(Image.new("RGB", (612, 792), "white"), dpi=72),
    )
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text=text,
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=50, t=700, r=550, b=650),
            charspan=(0, len(text)),
        ),
    )
    return doc


async def seed(config: AppConfig, name: str) -> dict[str, str]:
    """One paged document with a chunk and four stored pictures: PNG, JPEG,
    bytes no image library reads, and a truncated PNG."""
    dim = config.embeddings.model.vector_dim
    async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
        document = await rag.import_document(
            paged_document(f"{name} content"),
            [
                Chunk(
                    content=f"{name} content",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
                )
            ],
            uri=f"test://{name}/doc",
            title=f"{name} doc",
            metadata={"team": name},
        )
        assert document.id is not None
        [chunk] = await rag.chunk_repository.get_by_document_id(document.id)
        await rag.document_item_repository.create_items(
            document.id,
            [
                DocumentItem(
                    document_id=document.id,
                    position=100 + n,
                    self_ref=f"#/pictures/{n}",
                    label="picture",
                    picture_data=data,
                )
                for n, data in enumerate(
                    [
                        image_bytes("PNG"),
                        image_bytes("JPEG"),
                        b"not an image",
                        image_bytes("PNG")[:24],
                    ]
                )
            ],
        )
    assert chunk.id is not None
    return {"document_id": document.id, "chunk_id": chunk.id}


def two_databases(tmp_path) -> AppConfig:
    return AppConfig(
        lancedb=LanceDBConfig(
            databases={
                "alpha": str(tmp_path / "alpha.lancedb"),
                "beta": str(tmp_path / "beta.lancedb"),
                "missing": str(tmp_path / "missing.lancedb"),
            }
        )
    )


@pytest.fixture
def seed_database():
    return seed


@pytest.fixture
async def seeded(tmp_path):
    config = two_databases(tmp_path)
    ids = {name: await seed(config, name) for name in ("alpha", "beta")}
    return config, ids


@pytest.fixture
async def app(seeded):
    config, _ = seeded
    application = create_app(config)
    yield application
    await application.state.databases.aclose()


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as http:
        yield http
