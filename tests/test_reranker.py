import pytest

from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.reranking.vllm import VLLMReranker
from haiku.rag.store.models.chunk import Chunk

COHERE_AVAILABLE = bool(Config.COHERE_API_KEY)
VLLM_RERANK_AVAILABLE = bool(Config.VLLM_RERANK_BASE_URL)

chunks = [
    Chunk(content=content, document_id=str(i))
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


@pytest.mark.asyncio
async def test_reranker_base():
    reranker = RerankerBase()
    assert reranker._model == ""

    with pytest.raises(NotImplementedError):
        await reranker.rerank("query", [])


@pytest.mark.asyncio
async def test_mxbai_reranker():
    try:
        from haiku.rag.reranking.mxbai import MxBAIReranker

        Config.RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
        reranker = MxBAIReranker()
        # reranker._model = "mixedbread-ai/mxbai-rerank-base-v2"
        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert [chunk.document_id for chunk, score in reranked] == ["0", "2"]
        assert all(isinstance(score, float) for chunk, score in reranked)
        Config.RERANK_MODEL = ""

    except ImportError:
        pytest.skip("MxBAI package not installed")


@pytest.mark.asyncio
@pytest.mark.skipif(not COHERE_AVAILABLE, reason="Cohere API key not available")
async def test_cohere_reranker():
    try:
        from haiku.rag.reranking.cohere import CohereReranker

        reranker = CohereReranker()
        reranker._model = "rerank-v3.5"

        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert [chunk.document_id for chunk, score in reranked] == ["0", "2"]
        assert all(isinstance(score, float) for chunk, score in reranked)

    except ImportError:
        pytest.skip("Cohere package not installed")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not VLLM_RERANK_AVAILABLE, reason="vLLM rerank server not configured"
)
async def test_vllm_reranker():
    try:
        reranker = VLLMReranker("mixedbread-ai/mxbai-rerank-base-v2")

        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert [chunk.document_id for chunk, score in reranked] == ["0", "2"]
        assert all(isinstance(score, float) for chunk, score in reranked)

    except Exception:
        # Skip test if vLLM rerank server is not available
        pytest.skip("vLLM rerank server not available")
