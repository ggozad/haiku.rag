import pytest

from haiku.rag.reranking import get_reranker
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


@pytest.mark.asyncio
async def test_reranker_base():
    reranker = RerankerBase()
    assert reranker._model == "rerank-v3.5"

    with pytest.raises(NotImplementedError):
        await reranker.rerank("query", [])


@pytest.mark.asyncio
async def test_cohere_reranker():
    try:
        # Mock the client
        class MockResult:
            def __init__(self, index):
                self.index = index

        class MockResponse:
            def __init__(self, results):
                self.results = results

        class MockClient:
            def __init__(self, api_key=None):
                pass

            def rerank(self, model, query, documents, top_n):
                return MockResponse([MockResult(1), MockResult(0)])

        import haiku.rag.reranking.cohere

        original_client = haiku.rag.reranking.cohere.cohere.ClientV2
        haiku.rag.reranking.cohere.cohere.ClientV2 = MockClient

        try:
            from haiku.rag.reranking.cohere import CohereReranker

            reranker = CohereReranker()
            assert reranker._model == "rerank-v3.5"

            chunks = [
                Chunk(id=1, content="First chunk", document_id=1),
                Chunk(id=2, content="Second chunk", document_id=1),
            ]

            result = await reranker.rerank("test query", chunks)
            assert len(result) == 2
            assert result[0] == chunks[1]  # Should return chunk at index 1 first
            assert result[1] == chunks[0]  # Should return chunk at index 0 second
        finally:
            haiku.rag.reranking.cohere.cohere.ClientV2 = original_client

    except ImportError:
        pytest.skip("Cohere package not installed")


@pytest.mark.asyncio
async def test_get_reranker():
    try:

        class MockClient:
            def __init__(self, api_key=None):
                pass

        import haiku.rag.reranking.cohere

        original_client = haiku.rag.reranking.cohere.cohere.ClientV2
        haiku.rag.reranking.cohere.cohere.ClientV2 = MockClient

        try:
            reranker = get_reranker()
            assert reranker._model == "rerank-v3.5"
            assert hasattr(reranker, "rerank")
        finally:
            haiku.rag.reranking.cohere.cohere.ClientV2 = original_client
    except ImportError:
        pytest.skip("Cohere package not installed")
