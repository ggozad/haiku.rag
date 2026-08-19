from pathlib import Path
from unittest.mock import MagicMock

import pytest

from haiku.rag.config.models import AppConfig, ModelConfig, RerankingConfig
from haiku.rag.reranking import get_reranker
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.utils import raise_missing_extra

# Providers whose constructor loads a model in-process. Factory-routing tests
# patch the loader so they assert dispatch without paying the model load.
HEAVY_LOADERS = {
    "jina-local": "AutoModel",
    "cross-encoder": "CrossEncoder",
}


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
    reranker = RerankerBase()
    # The base carries no model: each reranker takes its own from the factory.
    assert reranker._model is None

    # Empty input short-circuits in the base class without dispatching to _rerank.
    assert await reranker.rerank("query", []) == []

    # The actual rerank step is abstract.
    with pytest.raises(NotImplementedError):
        await reranker.rerank("query", chunks)


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

    def test_unknown_provider_raises_error(self):
        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(provider="unknown_provider", name="some-model")
            )
        )
        with pytest.raises(ValueError, match="Unknown reranking provider"):
            get_reranker(config)

    def test_vllm_provider_without_base_url_raises_error(self):
        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(provider="vllm", name="BAAI/bge-reranker-v2-m3")
            )
        )
        with pytest.raises(ValueError, match="vLLM reranker requires base_url"):
            get_reranker(config)

    def test_cross_encoder_provider_without_name_raises_error(self):
        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(provider="cross-encoder", name="")
            )
        )
        with pytest.raises(ValueError, match="cross-encoder reranker requires name"):
            get_reranker(config)

    def test_multimodal_requires_vllm_provider(self):
        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(provider="cohere", name="rerank-v3.5"),
                multimodal=True,
            )
        )
        with pytest.raises(ValueError, match="multimodal"):
            get_reranker(config)

    def test_multimodal_vllm_provider_builds_reranker(self):
        pytest.importorskip("haiku.rag.reranking.vllm")
        from haiku.rag.reranking.vllm import VLLMReranker

        config = AppConfig(
            reranking=RerankingConfig(
                model=ModelConfig(
                    provider="vllm",
                    name="nvidia/llama-nemotron-rerank-vl-1b-v2",
                    base_url="http://localhost:8000",
                ),
                multimodal=True,
            )
        )
        assert isinstance(get_reranker(config), VLLMReranker)

    @pytest.mark.parametrize(
        "provider, model_name, class_module, class_name, extra_model_kwargs, expected_attrs, env_vars",
        [
            (
                "cohere",
                "rerank-english-v3.0",
                "haiku.rag.reranking.cohere",
                "CohereReranker",
                {},
                {"_model": "rerank-english-v3.0"},
                {"CO_API_KEY": "test-api-key"},
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
            (
                "cross-encoder",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "haiku.rag.reranking.cross_encoder",
                "CrossEncoderReranker",
                {},
                {"_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
                {},
            ),
        ],
        ids=[
            "cohere",
            "vllm",
            "zeroentropy",
            "zeroentropy-default",
            "jina",
            "jina-local",
            "cross-encoder",
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

        loader_attr = HEAVY_LOADERS.get(provider)
        if loader_attr:
            monkeypatch.setattr(mod, loader_attr, MagicMock())

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


class _PoolStats:
    """Fake httpx.AsyncClient factory counting constructions and closes."""

    def __init__(self, response_json):
        self.constructed = 0
        self.closed = 0
        stats = self

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return response_json

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                stats.constructed += 1

            async def post(self, url, json, headers):
                return FakeResponse()

            async def aclose(self):
                stats.closed += 1

        self.client_class = FakeAsyncClient


@pytest.mark.asyncio
async def test_vllm_reranker_reuses_pooled_client(monkeypatch):
    """One httpx client is built and reused across rerank calls; aclose
    releases it."""
    from haiku.rag.reranking.vllm import VLLMReranker

    stats = _PoolStats({"results": [{"index": 0, "relevance_score": 0.9}]})
    monkeypatch.setattr("httpx.AsyncClient", stats.client_class)

    reranker = VLLMReranker(model="m", base_url="http://localhost:8000")
    docs = [Chunk(content="a", order=0)]
    await reranker.rerank("q", docs)
    await reranker.rerank("q", docs)

    assert stats.constructed == 1
    await reranker.aclose()
    assert stats.closed == 1


@pytest.mark.asyncio
async def test_vllm_reranker_builds_multimodal_documents(monkeypatch):
    """Chunks carrying picture bytes are sent as content-parts documents
    (data-URI image plus text when the chunk has content); plain text chunks
    stay strings in the same request."""
    import base64

    from haiku.rag.reranking.vllm import VLLMReranker

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 2, "relevance_score": 0.7},
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, json, headers):
            captured["json"] = json
            return FakeResponse()

        async def aclose(self):
            pass

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    png = b"\x89PNG\r\n\x1a\n" + b"png-payload"
    jpeg = b"\xff\xd8\xff" + b"jpeg-payload"

    text_chunk = Chunk(content="plain text")
    described = Chunk(content="a described picture")
    described._picture_data = png
    undescribed = Chunk(content="")
    undescribed._picture_data = jpeg

    reranker = VLLMReranker(model="m", base_url="http://localhost:8000")
    reranked = await reranker.rerank("q", [text_chunk, described, undescribed])

    docs = captured["json"]["documents"]
    assert docs[0] == "plain text"

    image_part, text_part = docs[1]["content"]
    prefix = "data:image/png;base64,"
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith(prefix)
    assert base64.b64decode(image_part["image_url"]["url"].removeprefix(prefix)) == png
    assert text_part == {"type": "text", "text": "a described picture"}

    (jpeg_part,) = docs[2]["content"]
    assert jpeg_part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # Result indices pair back to the right chunks.
    assert [c for c, _ in reranked] == [text_chunk, described, undescribed]


@pytest.mark.asyncio
async def test_jina_reranker_reuses_pooled_client(monkeypatch):
    """One httpx client is built and reused across rerank calls; aclose
    releases it."""
    monkeypatch.setenv("JINA_API_KEY", "test-api-key")
    from haiku.rag.reranking.jina import JinaReranker

    stats = _PoolStats({"results": [{"index": 0, "relevance_score": 0.9}]})
    monkeypatch.setattr("httpx.AsyncClient", stats.client_class)

    reranker = JinaReranker("jina-reranker-v3")
    docs = [Chunk(content="a", order=0)]
    await reranker.rerank("q", docs)
    await reranker.rerank("q", docs)

    assert stats.constructed == 1
    await reranker.aclose()
    assert stats.closed == 1


@pytest.mark.asyncio
async def test_reranker_base_aclose_is_noop():
    """Base aclose exists so client teardown can close any reranker."""

    class Custom(RerankerBase):
        async def _rerank(self, query, chunks, top_n=10):
            return []

    await Custom().aclose()  # must not raise


def test_jina_reranker_missing_api_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    from haiku.rag.reranking.jina import JinaReranker

    with pytest.raises(ValueError, match="JINA_API_KEY environment variable required"):
        JinaReranker("jina-reranker-v3")


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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cross_encoder_reranker():
    try:
        from haiku.rag.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", chunks, top_n=2
        )
        assert len(reranked) == 2
        assert all(isinstance(score, float) for chunk, score in reranked)
        top_ids = [chunk.document_id for chunk, score in reranked]
        assert "0" in top_ids or "2" in top_ids
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cross_encoder_separates_saturated_scores():
    """`mxbai-rerank-base-v2` ships a Sigmoid it evaluates in bf16, where every
    strongly-relevant candidate rounds to exactly 1.0. Squashing the logits
    ourselves keeps them apart."""
    try:
        from haiku.rag.reranking.cross_encoder import CrossEncoderReranker

        saturating = [
            Chunk(content=content, document_id=str(i))
            for i, content in enumerate(
                [
                    "To Kill a Mockingbird is a novel by Harper Lee published in 1960.",
                    "Harper Lee wrote To Kill a Mockingbird, published in 1960.",
                    "The author of To Kill a Mockingbird is Harper Lee.",
                    "To Kill a Mockingbird, written by Harper Lee, appeared in 1960.",
                    "Harper Lee, author of To Kill a Mockingbird, won a Pulitzer Prize.",
                    "Harper Lee is the novelist who wrote To Kill a Mockingbird.",
                ]
            )
        ]

        reranker = CrossEncoderReranker("mixedbread-ai/mxbai-rerank-base-v2")
        reranked = await reranker.rerank(
            "Who wrote 'To Kill a Mockingbird'?", saturating, top_n=len(saturating)
        )

        scores = [score for chunk, score in reranked]
        assert len(set(scores)) == len(scores)
        assert all(0.0 < score < 1.0 for score in scores)
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.mark.asyncio
async def test_cross_encoder_reranks_via_model_ranking(monkeypatch):
    """The rank() results map back onto the input chunks by corpus_id."""
    import math

    import torch

    from haiku.rag.reranking import cross_encoder as ce_module

    class _StubCrossEncoder:
        def __init__(self, model):
            self.model = model

        def rank(self, query, documents, top_k=10, activation_fn=None):
            self.activation_fn = activation_fn
            # Reverse order so the mapping back to chunks is observable.
            return [
                {"corpus_id": i, "score": 1.0 - (i / 10)}
                for i in reversed(range(len(documents)))
            ][:top_k]

    monkeypatch.setattr(ce_module, "CrossEncoder", _StubCrossEncoder)

    reranker = ce_module.CrossEncoderReranker("stub/model")
    reranked = await reranker.rerank("query", chunks, top_n=2)

    assert len(reranked) == 2
    last_index = len(chunks) - 1
    assert reranked[0][0] is chunks[last_index]
    # The stub returns logits; the reranker squashes them itself.
    stub_logit = 1.0 - last_index / 10
    assert reranked[0][1] == pytest.approx(1.0 / (1.0 + math.exp(-stub_logit)))
    assert isinstance(reranker._reranker.activation_fn, torch.nn.Identity)


def test_missing_reranker_dependency_raises(monkeypatch):
    """A configured reranker whose extra is not installed must fail, not
    silently disable reranking."""
    import sys

    monkeypatch.setitem(sys.modules, "haiku.rag.reranking.zeroentropy", None)

    config = AppConfig(
        reranking=RerankingConfig(
            model=ModelConfig(provider="zeroentropy", name="zerank-1")
        )
    )

    with pytest.raises(ImportError):
        get_reranker(config)


def test_missing_extra_names_the_install_command():
    """The error tells the operator exactly what to install."""
    exc = ModuleNotFoundError("No module named 'cohere'", name="cohere")

    with pytest.raises(ImportError) as excinfo:
        raise_missing_extra("cohere", "cohere", exc)

    message = str(excinfo.value)
    assert "haiku.rag-slim[cohere]" in message
    assert excinfo.value.__cause__ is exc


def test_failure_inside_an_installed_package_is_not_reported_as_missing():
    """A broken transitive import must propagate untouched instead of claiming
    the package is not installed."""
    exc = ModuleNotFoundError("No module named 'torch._C'", name="torch._C")

    with pytest.raises(ModuleNotFoundError) as excinfo:
        raise_missing_extra("sentence_transformers", "cross-encoder", exc)

    assert excinfo.value is exc


def test_installed_reranker_extra_is_importable():
    """The guard must not fire for a dependency that is installed: the module
    imports and the reranker is constructible."""
    import haiku.rag.reranking.cohere as cohere_module

    assert cohere_module.CohereReranker is not None
