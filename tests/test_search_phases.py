import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.search import _rank
from haiku.rag.store.models import Chunk


@pytest.fixture
def exploding_reranker(monkeypatch):
    """A reranker that cannot be built, so any access fails loudly."""

    def boom(self):
        raise AssertionError("reranker built for a query that cannot use it")

    monkeypatch.setattr(HaikuRAG, "reranker", property(boom), raising=True)


@pytest.mark.asyncio
async def test_image_ranking_never_builds_the_reranker(
    temp_db_path, exploding_reranker
):
    """Local rerankers load model weights on construction, so an image query,
    which has no text to score against, must not touch one."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        candidates = [
            (Chunk(id="a", document_id="d", content="one"), 0.9),
            (Chunk(id="b", document_id="d", content="two"), 0.8),
        ]

        ranked = await _rank(rag, b"image-bytes", candidates, limit=1)

        assert [c.id for c, _ in ranked] == ["a"]


@pytest.mark.asyncio
async def test_image_fetch_never_builds_the_reranker(
    temp_db_path, exploding_reranker, monkeypatch
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        seen = {}

        async def fake_search(
            query, limit, search_type="hybrid", filter=None, query_vector=None
        ):
            seen["limit"] = limit
            return []

        async def fake_embed_image(self, image):
            return [0.1] * 8

        monkeypatch.setattr(rag.chunk_repository, "search", fake_search)
        monkeypatch.setattr(type(rag.embedder), "embed_image", fake_embed_image)
        monkeypatch.setattr(
            type(rag.embedder), "supports_images", property(lambda self: True)
        )

        await rag.search(b"image-bytes", limit=5)

        # No over-fetch: nothing will re-rank these.
        assert seen["limit"] == 5
