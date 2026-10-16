import base64
from io import BytesIO

import pytest
from PIL import Image

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk


@pytest.fixture
def query_embedding(monkeypatch, seeded):
    from haiku.rag.embeddings import EmbedderWrapper

    config, _ = seeded
    dim = config.embeddings.model.vector_dim

    async def embed(self, _query):
        return [0.1] * dim

    monkeypatch.setattr(EmbedderWrapper, "embed_query", embed)
    monkeypatch.setattr(EmbedderWrapper, "embed_image", embed)


@pytest.fixture
async def picture_chunk(seeded):
    """A chunk of the alpha document holding its first stored picture."""
    config, ids = seeded
    dim = config.embeddings.model.vector_dim
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        [chunk] = await rag.chunk_repository.create(
            [
                Chunk(
                    document_id=ids["alpha"]["document_id"],
                    content="A figure.",
                    embedding=[0.1] * dim,
                    order=1,
                    metadata={"doc_item_refs": ["#/pictures/0"]},
                )
            ]
        )
    return chunk.id


def _png(width: int = 8, height: int = 8) -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
async def two_pages(seeded):
    """A document of two pages: a chunked paragraph on the first, and on the
    second a paragraph no chunk holds and a captioned, described picture."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DescriptionMetaField,
        DoclingDocument,
        ImageRef,
        PictureMeta,
        ProvenanceItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    def prov(page: int) -> ProvenanceItem:
        return ProvenanceItem(
            page_no=page, bbox=BoundingBox(l=50, t=700, r=550, b=650), charspan=(0, 4)
        )

    doc = DoclingDocument(name="two pages")
    for page in (1, 2):
        doc.add_page(
            page_no=page,
            size=Size(width=612.0, height=792.0),
            image=ImageRef.from_pil(Image.new("RGB", (612, 792), "white"), dpi=72),
        )
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="first", prov=prov(1))
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="second", prov=prov(2))
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Figure 1.", prov=prov(2))
    picture = doc.add_picture(caption=caption, prov=prov(2))
    picture.meta = PictureMeta(description=DescriptionMetaField(text="A red square."))

    config, _ = seeded
    dim = config.embeddings.model.vector_dim
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        document = await rag.import_document(
            doc,
            [
                Chunk(
                    content="first",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
                ),
                Chunk(
                    content="A red square.",
                    embedding=[0.1] * dim,
                    order=1,
                    metadata={"doc_item_refs": ["#/pictures/0"], "page_numbers": [2]},
                ),
            ],
            uri="test://alpha/two-pages",
            title="two pages",
        )
        chunks = await rag.chunk_repository.get_by_document_id(document.id)
    return document.id, [c.id for c in chunks]


async def test_a_page_no_chunk_draws_on_is_listed_and_shown(client, two_pages):
    document_id, (first, picture) = two_pages
    base = f"/api/inspect/alpha/documents/{document_id}/pages"
    pages = (await client.get(base)).json()["pages"]
    assert pages == [
        {"page": 1, "chunk_ids": [first]},
        {"page": 2, "chunk_ids": [picture]},
    ]
    image = await client.get(f"{base}/2")
    assert image.content.startswith(b"\x89PNG")


async def test_a_page_holding_only_unchunked_items_is_shown_plain(
    client, two_pages, seeded
):
    document_id, (first, picture) = two_pages
    config, _ = seeded
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        await rag.store.chunks_table.delete(f"id = '{picture}'")
    base = f"/api/inspect/alpha/documents/{document_id}/pages"
    pages = (await client.get(base)).json()["pages"]
    assert pages == [{"page": 1, "chunk_ids": [first]}, {"page": 2, "chunk_ids": []}]
    image = await client.get(f"{base}/2")
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")


async def test_a_plain_page_without_an_image_says_so(
    client, two_pages, seeded, monkeypatch
):
    from haiku.rag.store.models.document import Document

    document_id, (_, picture) = two_pages
    config, _ = seeded
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        await rag.store.chunks_table.delete(f"id = '{picture}'")
    monkeypatch.setattr(Document, "get_page_images", lambda self, pages: {})
    response = await client.get(f"/api/inspect/alpha/documents/{document_id}/pages/2")
    assert response.status_code == 404


async def test_pictures_carry_caption_description_and_chunk(client, two_pages):
    document_id, (_, picture) = two_pages
    body = (
        await client.get(f"/api/inspect/alpha/documents/{document_id}/pictures")
    ).json()
    assert body["pictures"] == [
        {
            "self_ref": "#/pictures/0",
            "caption": "Figure 1.",
            "description": "A red square.",
            "page_numbers": [2],
            "chunk_id": picture,
        }
    ]
    missing = await client.get("/api/inspect/alpha/documents/nope/pictures")
    assert missing.status_code == 404


async def test_a_chunk_is_rendered_as_the_agent_receives_it(client, seeded):
    _, ids = seeded
    body = (
        await client.get(
            f"/api/inspect/alpha/chunks/{ids['alpha']['chunk_id']}/context"
        )
    ).json()
    assert "alpha content" in body["rendering"]
    assert ids["alpha"]["document_id"] in body["rendering"]
    assert body["pictures"] == []
    assert body["document_id"] == ids["alpha"]["document_id"]


async def test_a_chunk_holding_a_picture_shows_it(client, picture_chunk):
    body = (
        await client.get(f"/api/inspect/alpha/chunks/{picture_chunk}/context")
    ).json()
    assert body["pictures"] == ["#/pictures/0"]


async def test_similar_chunks_leave_the_chunk_out(
    client, picture_chunk, query_embedding
):
    body = (
        await client.get(f"/api/inspect/alpha/chunks/{picture_chunk}/similar")
    ).json()
    ids = [row["chunk_id"] for row in body["results"]]
    assert picture_chunk not in ids
    assert ids
    assert all(row["source"] == "alpha" for row in body["results"])
    missing = await client.get("/api/inspect/alpha/chunks/nope/similar")
    assert missing.status_code == 404


async def test_pages_list_their_chunks_and_draw_them(client, seeded):
    _, ids = seeded
    base = f"/api/inspect/beta/documents/{ids['beta']['document_id']}/pages"
    pages = (await client.get(base)).json()["pages"]
    assert pages == [{"page": 1, "chunk_ids": [ids["beta"]["chunk_id"]]}]
    image = await client.get(f"{base}/1")
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    assert (await client.get(f"{base}/2")).status_code == 404
    assert (await client.get(f"{base}/one")).status_code == 400


async def test_a_page_without_an_image_says_so(client, seeded, monkeypatch):
    _, ids = seeded

    async def no_images(self, *args, **kwargs):
        return []

    monkeypatch.setattr(HaikuRAG, "visualize_chunk", no_images)
    response = await client.get(
        f"/api/inspect/beta/documents/{ids['beta']['document_id']}/pages/1"
    )
    assert response.status_code == 404


async def test_search_spans_the_selected_databases(client, query_embedding):
    body = (
        await client.post(
            "/api/search",
            json={"query": "content", "sources": ["alpha", "beta"], "limit": 5},
        )
    ).json()
    assert {row["source"] for row in body["results"]} == {"alpha", "beta"}
    assert "Collection: alpha" in body["rendering"]
    assert "[rank 1 of" in body["rendering"]


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {"query": "  "},
        {"query": "x", "limit": 0},
        {"query": "x", "search_type": "fuzzy"},
        {"query": "x", "sources": "alpha"},
        [],
        None,
    ],
)
async def test_a_malformed_search_is_refused(client, request_body):
    response = await client.post("/api/search", json=request_body)
    assert response.status_code == 400
    invalid = await client.post("/api/search", content=b"{not json")
    assert invalid.status_code == 400


async def test_a_failing_search_says_why(client, query_embedding):
    response = await client.post(
        "/api/search",
        json={"query": "x", "sources": ["alpha"], "filter": "nonsense ==="},
    )
    assert response.status_code == 400
    assert response.json()["error"].startswith("the search failed:")


async def test_an_unknown_database_in_a_search_is_named(client, query_embedding):
    response = await client.post(
        "/api/search", json={"query": "x", "sources": ["gamma"]}
    )
    assert response.status_code == 400
    assert "gamma" in response.json()["error"]


async def test_an_image_search_needs_an_embedder_that_embeds_images(client):
    image = base64.b64encode(b"\x89PNG").decode()
    response = await client.post("/api/search", json={"image": image})
    assert response.status_code == 400
    assert response.json() == {"error": "the embedder does not embed images"}


async def test_an_image_search_runs_on_a_multimodal_embedder(
    client, query_embedding, monkeypatch
):
    from haiku.rag.embeddings import EmbedderWrapper

    monkeypatch.setattr(EmbedderWrapper, "supports_images", True)
    response = await client.post(
        "/api/search", json={"image": _png(), "sources": ["alpha"]}
    )
    assert response.status_code == 200
    assert response.json()["results"]


@pytest.mark.parametrize(
    ("image", "reason"),
    [
        ("not base64!", "base64"),
        (5, "base64"),
        ("é", "base64"),
        (base64.b64encode(b"hello").decode(), "not an image"),
        (_png()[: len(_png()) // 2 // 4 * 4], "not an image"),
        (_png(2049, 1), "2048"),
    ],
)
async def test_an_unacceptable_search_image_is_refused(
    client, query_embedding, monkeypatch, image, reason
):
    from haiku.rag.embeddings import EmbedderWrapper

    monkeypatch.setattr(EmbedderWrapper, "supports_images", True)
    response = await client.post("/api/search", json={"image": image})
    assert response.status_code == 400
    assert reason in response.json()["error"]


async def test_info_reports_tables_tags_and_history(client, seeded):
    config, _ = seeded
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        await rag.store.create_tag("before-review")
    body = (await client.get("/api/inspect/alpha/info")).json()
    assert body["info"]["exists"] is True
    assert {t["name"] for t in body["info"]["tables"]} >= {"documents", "chunks"}
    assert body["tags"]["before-review"]["complete"] is True
    newest = body["history"]["chunks"][0]
    assert "before-review" in newest["tags"]


async def test_info_reports_a_database_the_client_refuses(client):
    body = (await client.get("/api/inspect/missing/info")).json()
    assert body["info"]["exists"] is False
    assert body["tags"] is None
    assert "missing" in body["unavailable"]
    assert (await client.get("/api/inspect/gamma/info")).status_code == 404


async def _info(config, name: str, sessions_path) -> dict:
    """The info route of an app built over `config`."""
    import httpx

    from haiku_rag_app.main import create_app

    app = create_app(config, sessions_path=sessions_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            return (await http.get(f"/api/inspect/{name}/info")).json()
    finally:
        await app.state.databases.aclose()
        await app.state.sessions.aclose()


async def test_info_reads_a_configured_tag(tmp_path, seeded):
    from haiku.rag.config.models import AppConfig, DatabaseEntry, LanceDBConfig

    config, _ = seeded
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        await rag.store.create_tag("frozen")
    location = config.lancedb.databases["alpha"]
    tagged = AppConfig(
        lancedb=LanceDBConfig(
            databases={"frozen": DatabaseEntry(location=location, tag="frozen")}
        )
    )
    body = await _info(tagged, "frozen", tmp_path / "tagged.db")
    assert body["info"]["exists"] is True
    assert "frozen" in body["tags"]


async def test_info_reports_an_existing_database_the_client_refuses(tmp_path, seeded):
    from haiku.rag.config.models import AppConfig

    config, _ = seeded
    model = config.embeddings.model.model_dump()
    model["vector_dim"] += 1
    mismatched = AppConfig.model_validate(
        {
            "lancedb": {"databases": {"alpha": config.lancedb.databases["alpha"]}},
            "embeddings": {"model": model},
        }
    )
    body = await _info(mismatched, "alpha", tmp_path / "mismatched.db")
    assert body["info"]["exists"] is True
    assert body["tags"] is None
    assert body["unavailable"]


async def test_an_image_pillow_rejects_while_decoding_is_refused(
    client, query_embedding, monkeypatch
):
    from haiku.rag.embeddings import EmbedderWrapper

    def broken(self):
        raise ValueError("truncated sRGB chunk")

    image = _png()
    monkeypatch.setattr(EmbedderWrapper, "supports_images", True)
    monkeypatch.setattr("PIL.PngImagePlugin.PngImageFile.load", broken)
    response = await client.post("/api/search", json={"image": image})
    assert response.status_code == 400
    assert "not an image" in response.json()["error"]


async def test_a_search_body_over_the_limit_is_refused(client, monkeypatch):
    from haiku_rag_app import inspect

    monkeypatch.setattr(inspect, "MAX_SEARCH_BODY", 16)

    async def chunks():
        yield b'{"query": '
        yield b'"' + b"x" * 64 + b'"}'

    response = await client.post("/api/search", content=chunks())
    assert response.status_code == 413
