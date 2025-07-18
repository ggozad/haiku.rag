import pytest

from haiku.rag.reranking import get_reranker
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.reranking.mxbai import MxBAIReranker
from haiku.rag.store.models.chunk import Chunk


@pytest.mark.asyncio
async def test_reranker_base():
    reranker = RerankerBase()
    assert reranker._model == "mixedbread-ai/mxbai-rerank-base-v2"

    with pytest.raises(NotImplementedError):
        await reranker.rerank("query", [])


@pytest.mark.asyncio
async def test_mxbai_reranker():
    reranker = MxBAIReranker()
    chunks = [
        Chunk(content=content, document_id=i)
        for i, content in enumerate(
            [
                "To Kill a Mockingbird is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
                "The novel Moby-Dick was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
                "Harper Lee, an American novelist widely known for her novel To Kill a Mockingbird, was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
                "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
                "The Harry Potter series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
                "The Great Gatsby, a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan.",
            ]
        )
    ]

    reranked = await reranker.rerank(
        "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
    )
    assert [r.document_id for r in reranked] == [0, 2]


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
