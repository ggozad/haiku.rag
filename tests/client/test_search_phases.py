import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.search import _rank
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import Chunk


@pytest.fixture
def exploding_reranker(monkeypatch):
    """A reranker that cannot be built, so any access fails loudly."""

    def boom(self):
        raise AssertionError("reranker built for a query that cannot use it")

    monkeypatch.setattr(HaikuRAG, "reranker", property(boom), raising=True)


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


async def test_an_image_query_ignores_a_full_text_search_type(
    temp_db_path, monkeypatch
):
    """An image has no text to match, so it searches by vector whatever the
    caller asked for."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        seen = {}

        async def fake_search(query, limit, search_type, filter, query_vector):
            seen.update({"search_type": search_type, "query_vector": query_vector})
            return []

        async def fake_embed_image(self, image):
            return [0.1] * 8

        monkeypatch.setattr(rag.chunk_repository, "search", fake_search)
        monkeypatch.setattr(type(rag.embedder), "embed_image", fake_embed_image)
        monkeypatch.setattr(
            type(rag.embedder), "supports_images", property(lambda self: True)
        )

        await rag.search(b"image-bytes", search_type="fts")

        assert seen["search_type"] == "vector"
        assert seen["query_vector"] == [0.1] * 8


class _EmptyReranker:
    async def rerank(self, query, chunks, top_n=10):
        return []


@pytest.mark.parametrize("factor", [10, 20])
async def test_a_reranked_text_search_fetches_the_configured_pool(
    temp_db_path, monkeypatch, factor
):
    config = AppConfig()
    config.search.rerank_pool_factor = factor
    monkeypatch.setattr(HaikuRAG, "reranker", property(lambda self: _EmptyReranker()))
    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        seen = {}

        async def fake_search(
            query, limit, search_type="hybrid", filter=None, query_vector=None
        ):
            seen["limit"] = limit
            return []

        monkeypatch.setattr(rag.chunk_repository, "search", fake_search)

        await rag.search("text", limit=5)

    assert seen["limit"] == 5 * factor
