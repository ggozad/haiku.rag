from pathlib import Path

import pytest

from haiku.rag.config import get_config


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Cassettes sit with the rest, under `tests/cassettes/multi_db/`."""
    module = request.module.__name__.rsplit(".", 1)[-1]
    return str(Path(__file__).parent.parent / "cassettes" / "multi_db" / module)


@pytest.fixture
def query_embedding(monkeypatch):
    """Vector search with no embedder behind it, recording the queries embedded.

    These tests are about which databases are asked and how often, not about
    retrieval quality, and CI has no embedding endpoint.
    """
    from haiku.rag.embeddings import EmbedderWrapper

    embedded: list[str] = []

    async def embed_query(self, text):
        embedded.append(text)
        return [0.1] * get_config().embeddings.model.vector_dim

    monkeypatch.setattr(EmbedderWrapper, "embed_query", embed_query)
    return embedded
