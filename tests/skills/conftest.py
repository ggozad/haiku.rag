import random
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.skills._deps import AnalysisRunDeps, RAGRunDeps

VECTOR_DIM = 2560


def _make_ctx(state=None, rag=None):
    """Create a mock RunContext with RAGRunDeps (or AnalysisRunDeps when state is AnalysisState)."""
    from haiku.rag.skills.analysis import AnalysisState

    ctx = MagicMock(spec=RunContext)
    if isinstance(state, AnalysisState):
        ctx.deps = AnalysisRunDeps(state=state, rag=rag)
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

    async def fake_embed_query(self, text):
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(VECTOR_DIM)]

    async def fake_embed_documents(self, texts):
        result = []
        for t in texts:
            random.seed(hash(t) % (2**32))
            result.append([random.random() for _ in range(VECTOR_DIM)])
        return result

    monkeypatch.setattr(EmbedderWrapper, "embed_query", fake_embed_query)
    monkeypatch.setattr(EmbedderWrapper, "embed_documents", fake_embed_documents)


@pytest.fixture
def test_app_config():
    return AppConfig(environment="skills-test")


@pytest.fixture
async def rag_db(temp_db_path):
    """Create a test database with sample documents."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
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
    return temp_db_path


@pytest.fixture
async def rag_client(rag_db):
    """Yield an open read-only HaikuRAG client on the sample db."""
    async with HaikuRAG(rag_db, read_only=True) as rag:
        yield rag
