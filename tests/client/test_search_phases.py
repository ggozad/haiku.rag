import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig


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
