import random
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.skills._deps import AnalysisRunDeps, RAGRunDeps

VECTOR_DIM = 2560


def _seeded_vector(text: str) -> list[float]:
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(VECTOR_DIM)]


async def _fake_embed_query(self, text: str) -> list[float]:
    return _seeded_vector(text)


async def _fake_embed_documents(self, texts: list[str]) -> list[list[float]]:
    return [_seeded_vector(t) for t in texts]


def _make_ctx(state=None, rag=None, sandbox=None):
    """Create a mock RunContext with RAGRunDeps (or AnalysisRunDeps when state is AnalysisState)."""
    from haiku.rag.skills.analysis import AnalysisState

    ctx = MagicMock(spec=RunContext)
    if isinstance(state, AnalysisState) or sandbox is not None:
        ctx.deps = AnalysisRunDeps(state=state, rag=rag, sandbox=sandbox)
    else:
        ctx.deps = RAGRunDeps(state=state, rag=rag)
    return ctx


def _get_tool(skill, name):
    """Get a tool function from a skill by name."""
    for tool in skill.tools:
        if callable(tool) and tool.__name__ == name:
            return tool
    raise ValueError(f"Tool {name!r} not found in skill")


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Monkeypatch the embedder to return deterministic vectors."""
    monkeypatch.setattr(EmbedderWrapper, "embed_query", _fake_embed_query)
    monkeypatch.setattr(EmbedderWrapper, "embed_documents", _fake_embed_documents)


@pytest.fixture
def test_app_config():
    return AppConfig(environment="skills-test")


@pytest.fixture(scope="session")
async def rag_db(tmp_path_factory):
    """Sample database with two documents, built once and shared read-only.

    Consumers (``rag_client``, ``sandbox_factory``) only read, so the docling
    conversion + ingest is paid once per session instead of per test. Document
    vectors use the same seeded fakes as ``mock_embedder`` so search stays
    consistent with query-time embeddings.
    """
    db_path = tmp_path_factory.mktemp("skills_rag_db") / "rag.lancedb"
    with (
        patch.object(EmbedderWrapper, "embed_query", _fake_embed_query),
        patch.object(EmbedderWrapper, "embed_documents", _fake_embed_documents),
    ):
        async with HaikuRAG(db_path, create=True) as rag:
            await rag.create_document(
                "Artificial intelligence is transforming industries worldwide. "
                "Deep learning models are used in healthcare, finance, and transportation.",
                title="AI Overview",
                uri="test://ai-overview",
            )
            await rag.create_document(
                "Machine learning is a subset of artificial intelligence. "
                "It includes supervised learning, unsupervised learning, and reinforcement learning.",
                title="ML Basics",
                uri="test://ml-basics",
            )
    return db_path


@pytest.fixture
async def rag_client(rag_db):
    """Yield an open read-only HaikuRAG client on the sample db."""
    async with HaikuRAG(rag_db, read_only=True) as rag:
        yield rag


@pytest.fixture
def sandbox_factory(rag_db, test_app_config):
    """Build Sandbox instances bound to the sample db, optionally with a doc filter."""
    from haiku.rag.sandbox import AnalysisContext, Sandbox

    def _make(filter: str | None = None) -> Sandbox:
        return Sandbox(
            db_path=rag_db,
            config=test_app_config,
            context=AnalysisContext(filter=filter),
        )

    return _make
