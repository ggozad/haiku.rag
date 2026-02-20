import random
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.embeddings import EmbedderWrapper
from haiku.skills.state import SkillRunDeps

VECTOR_DIM = 2560


def _make_ctx(state=None):
    """Create a mock RunContext with SkillRunDeps."""
    ctx = MagicMock(spec=RunContext)
    ctx.deps = SkillRunDeps(state=state)
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
