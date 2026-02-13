from pathlib import Path

import pytest

from haiku.rag.config.models import AppConfig, ModelConfig, RerankingConfig
from haiku.rag.reranking import get_reranker
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_reranker")


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
    from haiku.rag.config import Config

    reranker = RerankerBase()
    expected_model = Config.reranking.model.name if Config.reranking.model else None
    assert reranker._model == expected_model

    with pytest.raises(NotImplementedError):
        await reranker.rerank("query", [])


@pytest.mark.asyncio
async def test_mxbai_reranker():
    try:
        from haiku.rag.config import Config
        from haiku.rag.config.models import ModelConfig
        from haiku.rag.reranking.mxbai import MxBAIReranker

        Config.reranking.model = ModelConfig(
            provider="mxbai", name="mixedbread-ai/mxbai-rerank-base-v2"
        )
        reranker = MxBAIReranker()
        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert [chunk.document_id for chunk, score in reranked] == ["0", "2"]
        assert all(isinstance(score, float) for chunk, score in reranked)
        Config.reranking.model = None

    except ImportError:
        pytest.skip("MxBAI package not installed")


@pytest.mark.asyncio
@pytest.mark.vcr()
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
@pytest.mark.vcr()
async def test_zeroentropy_reranker():
    try:
        from haiku.rag.reranking.zeroentropy import ZeroEntropyReranker

        reranker = ZeroEntropyReranker("zerank-1")

        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert len(reranked) == 2
        assert all(isinstance(score, float) for chunk, score in reranked)
        # Check that the top results are relevant to Harper Lee / To Kill a Mockingbird
        top_ids = [chunk.document_id for chunk, score in reranked]
        assert "0" in top_ids or "2" in top_ids  # These chunks mention the book/author

    except ImportError:
        pytest.skip("Zero Entropy package not installed")


class TestGetReranker:
    def test_returns_none_when_no_model_configured(self):
        config = AppConfig(reranking=RerankingConfig(model=None))
        result = get_reranker(config)
        assert result is None

    def test_unknown_provider_returns_none(self):
        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(provider="unknown_provider", name="some-model")
            )
        )
        result = get_reranker(config)
        assert result is None

    def test_vllm_provider_without_base_url_raises_error(self):
        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(provider="vllm", name="BAAI/bge-reranker-v2-m3")
            )
        )
        with pytest.raises(ValueError, match="vLLM reranker requires base_url"):
            get_reranker(config)

    @pytest.mark.parametrize(
        "provider, model_name, class_module, class_name, extra_model_kwargs, expected_attrs, env_vars",
        [
            (
                "mxbai",
                "mixedbread-ai/mxbai-rerank-base-v2",
                "haiku.rag.reranking.mxbai",
                "MxBAIReranker",
                {},
                {},
                {},
            ),
            (
                "cohere",
                "rerank-v3.5",
                "haiku.rag.reranking.cohere",
                "CohereReranker",
                {},
                {},
                {},
            ),
            (
                "vllm",
                "BAAI/bge-reranker-v2-m3",
                "haiku.rag.reranking.vllm",
                "VLLMReranker",
                {"base_url": "http://localhost:8000"},
                {
                    "_model": "BAAI/bge-reranker-v2-m3",
                    "_base_url": "http://localhost:8000",
                },
                {},
            ),
            (
                "zeroentropy",
                "zerank-1",
                "haiku.rag.reranking.zeroentropy",
                "ZeroEntropyReranker",
                {},
                {"_model": "zerank-1"},
                {},
            ),
            (
                "zeroentropy",
                "",
                "haiku.rag.reranking.zeroentropy",
                "ZeroEntropyReranker",
                {},
                {"_model": "zerank-1"},
                {},
            ),
            (
                "jina",
                "jina-reranker-v3",
                "haiku.rag.reranking.jina",
                "JinaReranker",
                {},
                {"_model": "jina-reranker-v3"},
                {"JINA_API_KEY": "test-api-key"},
            ),
            (
                "jina-local",
                "jinaai/jina-reranker-v3",
                "haiku.rag.reranking.jina_local",
                "JinaLocalReranker",
                {},
                {"_model": "jinaai/jina-reranker-v3"},
                {},
            ),
        ],
        ids=[
            "mxbai",
            "cohere",
            "vllm",
            "zeroentropy",
            "zeroentropy-default",
            "jina",
            "jina-local",
        ],
    )
    def test_provider(
        self,
        provider,
        model_name,
        class_module,
        class_name,
        extra_model_kwargs,
        expected_attrs,
        env_vars,
        monkeypatch,
    ):
        mod = pytest.importorskip(class_module)
        expected_class = getattr(mod, class_name)

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(
                    provider=provider, name=model_name, **extra_model_kwargs
                )
            )
        )
        result = get_reranker(config)
        assert isinstance(result, expected_class)

        for attr, value in expected_attrs.items():
            assert getattr(result, attr) == value


def test_jina_reranker_missing_api_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    from haiku.rag.reranking.jina import JinaReranker

    with pytest.raises(ValueError, match="JINA_API_KEY environment variable required"):
        JinaReranker("jina-reranker-v3")


@pytest.mark.asyncio
async def test_jina_reranker_empty_chunks(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-api-key")

    from haiku.rag.reranking.jina import JinaReranker

    reranker = JinaReranker("jina-reranker-v3")
    result = await reranker.rerank("query", [], top_n=2)
    assert result == []


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_jina_reranker(monkeypatch):
    import os

    # Only set dummy key if real key not present (for VCR playback)
    if not os.environ.get("JINA_API_KEY"):
        monkeypatch.setenv("JINA_API_KEY", "test-api-key")

    from haiku.rag.reranking.jina import JinaReranker

    reranker = JinaReranker("jina-reranker-v3")

    reranked = await reranker.rerank(
        "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
    )
    assert len(reranked) == 2
    assert all(isinstance(score, float) for chunk, score in reranked)
    # Check that the top results are relevant to Harper Lee / To Kill a Mockingbird
    top_ids = [chunk.document_id for chunk, score in reranked]
    assert "0" in top_ids or "2" in top_ids  # These chunks mention the book/author


@pytest.mark.asyncio
@pytest.mark.integration
async def test_jina_local_reranker():
    try:
        from haiku.rag.reranking.jina_local import JinaLocalReranker

        reranker = JinaLocalReranker("jinaai/jina-reranker-v3")

        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert len(reranked) == 2
        assert all(isinstance(score, float) for chunk, score in reranked)
        # Check that the top results are relevant to Harper Lee / To Kill a Mockingbird
        top_ids = [chunk.document_id for chunk, score in reranked]
        assert "0" in top_ids or "2" in top_ids  # These chunks mention the book/author
    except ImportError:
        pytest.skip("Jina local dependencies not installed")
