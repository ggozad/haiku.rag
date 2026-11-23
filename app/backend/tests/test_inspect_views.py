import base64
from datetime import UTC, datetime, timedelta

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk
from tests.support import letter_pdf, png, served


@pytest.fixture
def query_embedding(query_embedding, monkeypatch, seeded):
    """Also embeds an image query."""
    from haiku.rag.embeddings import EmbedderWrapper

    config, _ = seeded
    dim = config.embeddings.model.vector_dim

    async def embed_image(self, _image):
        return [0.1] * dim

    monkeypatch.setattr(EmbedderWrapper, "embed_image", embed_image)


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


@pytest.fixture
async def two_pages(seeded):
    """A document of two pages: a chunked paragraph on the first, and on the
    second a paragraph no chunk holds and a captioned, described picture."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DescriptionMetaField,
        DoclingDocument,
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
        doc.add_page(page_no=page, size=Size(width=612.0, height=792.0))
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
            original=letter_pdf(2),
        )
        chunks = await rag.chunk_repository.get_by_document_id(document.id)
    return document.id, [c.id for c in chunks]


async def test_pages_list_their_chunks_and_the_items_they_hold(client, two_pages):
    document_id, (first, picture) = two_pages
    base = f"/api/inspect/alpha/documents/{document_id}/pages"
    pages = (await client.get(base)).json()["pages"]
    assert pages == [
        {"page": 1, "chunk_ids": [first]},
        {"page": 2, "chunk_ids": [picture]},
    ]


async def test_a_page_draws_the_boxes_of_every_chunk_on_it(client, two_pages):
    document_id, (first, picture) = two_pages
    base = f"/api/inspect/alpha/documents/{document_id}/pages"

    second = (await client.get(f"{base}/2/boxes")).json()
    assert second["document_id"] == document_id
    assert [page["page"] for page in second["pages"]] == [2]
    assert all(box["matched"] for box in second["pages"][0]["boxes"])

    empty = (await client.get(f"{base}/3/boxes")).json()
    assert (empty["pages"], empty["message"]) == ([], "No visual grounding available")
    assert (await client.get(f"{base}/two/boxes")).status_code == 400


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
        {"json": {}},
        {"json": {"query": "  "}},
        {"json": {"query": "x", "limit": 0}},
        {"json": {"query": "x", "search_type": "fuzzy"}},
        {"json": {"query": "x", "sources": "alpha"}},
        {"json": []},
        {"json": None},
        {"content": b"{not json"},
    ],
)
async def test_a_malformed_search_is_refused(client, request_body):
    response = await client.post("/api/search", **request_body)
    assert response.status_code == 400


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
        "/api/search", json={"image": png(), "sources": ["alpha"]}
    )
    assert response.status_code == 200
    assert response.json()["results"]


async def test_an_unacceptable_search_image_is_refused(
    client, query_embedding, monkeypatch
):
    from haiku.rag.embeddings import EmbedderWrapper

    monkeypatch.setattr(EmbedderWrapper, "supports_images", True)
    response = await client.post("/api/search", json={"image": "not base64!"})
    assert response.status_code == 400
    assert "base64" in response.json()["error"]


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
    written = datetime.fromisoformat(newest["timestamp"])
    assert abs(datetime.now(UTC) - written) < timedelta(minutes=1)


async def test_info_reports_a_database_the_client_refuses(client):
    body = (await client.get("/api/inspect/missing/info")).json()
    assert body["info"]["exists"] is False
    assert body["tags"] is None
    assert "missing" in body["unavailable"]
    assert (await client.get("/api/inspect/gamma/info")).status_code == 404


async def _info(config, name: str, sessions_path) -> dict:
    """The info route of an app built over `config`."""
    async with served(config, sessions_path) as (_, http):
        return (await http.get(f"/api/inspect/{name}/info")).json()


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


async def test_a_search_body_over_the_limit_is_refused(client, monkeypatch):
    from haiku_rag_app import inspect

    monkeypatch.setattr(inspect, "MAX_SEARCH_BODY", 16)

    async def chunks():
        yield b'{"query": '
        yield b'"' + b"x" * 64 + b'"}'

    response = await client.post("/api/search", content=chunks())
    assert response.status_code == 413
