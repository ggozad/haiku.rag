import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic_ai import format_as_xml

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult

if TYPE_CHECKING:
    from haiku.rag.embeddings import EmbedderWrapper

MAX_QA_HISTORY = 50

_embedding_cache: dict[str, list[float]] = {}


def _qa_cache_key(question: str, answer: str) -> str:
    """Generate cache key from Q/A content."""
    return hashlib.sha256(f"Q: {question}\nA: {answer}".encode()).hexdigest()


class CitationInfo(BaseModel):
    """Citation info for frontend display."""

    index: int
    document_id: str
    chunk_id: str
    document_uri: str
    document_title: str | None = None
    page_numbers: list[int] = []
    headings: list[str] | None = None
    content: str


class QAResponse(BaseModel):
    """A Q&A pair from conversation history with citations."""

    question: str
    answer: str
    confidence: float = 0.9
    citations: list[CitationInfo] = []

    @property
    def sources(self) -> list[str]:
        """Source names for display."""
        return list(
            dict.fromkeys(c.document_title or c.document_uri for c in self.citations)
        )


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    session_id: str = ""
    citations: list[CitationInfo] = []
    qa_history: list[QAResponse] = []


def format_conversation_context(qa_history: list[QAResponse]) -> str:
    """Format conversation history as XML for inclusion in prompts."""
    if not qa_history:
        return ""

    context_data = {
        "previous_qa": [
            {
                "question": qa.question,
                "answer": qa.answer,
                "sources": qa.sources,
            }
            for qa in qa_history
        ],
    }
    return format_as_xml(context_data, root_tag="conversation_context")


def _cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def rank_qa_history_by_similarity(
    current_question: str,
    qa_history: list[QAResponse],
    embedder: "EmbedderWrapper",
    top_k: int = 5,
) -> list[QAResponse]:
    """Rank Q&A history by semantic similarity to current question.

    Embeds question+answer pairs and returns the top-K most similar to the
    current question. Falls back to returning the last top_k entries if
    embedding fails.

    Args:
        current_question: The current question to compare against.
        qa_history: List of previous Q&A pairs.
        embedder: Embedder instance to use for embedding.
        top_k: Maximum number of entries to return.

    Returns:
        Top-K Q&A pairs ranked by similarity to current question.
    """
    if not qa_history:
        return []

    if len(qa_history) <= top_k:
        return qa_history

    # Embed current question
    question_embedding = np.array(await embedder.embed_query(current_question))

    # Check cache and collect uncached entries
    qa_embeddings: list[list[float]] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, qa in enumerate(qa_history):
        cache_key = _qa_cache_key(qa.question, qa.answer)
        if cache_key in _embedding_cache:
            qa_embeddings.append(_embedding_cache[cache_key])
        else:
            qa_embeddings.append([])  # placeholder
            uncached_indices.append(i)
            uncached_texts.append(f"Q: {qa.question}\nA: {qa.answer}")

    # Embed only uncached entries
    if uncached_texts:
        new_embeddings = await embedder.embed_documents(uncached_texts)
        for idx, embedding in zip(uncached_indices, new_embeddings):
            qa = qa_history[idx]
            cache_key = _qa_cache_key(qa.question, qa.answer)
            _embedding_cache[cache_key] = embedding
            qa_embeddings[idx] = embedding

    # Compute similarities
    similarities: list[tuple[int, float]] = []
    for i, qa_emb in enumerate(qa_embeddings):
        sim = _cosine_similarity(question_embedding, np.array(qa_emb))
        similarities.append((i, sim))

    # Sort by similarity (descending) and take top-K
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in similarities[:top_k]])

    # Return in original order
    return [qa_history[i] for i in top_indices]


@dataclass
class ChatDeps:
    """Dependencies for chat agent."""

    client: HaikuRAG
    config: AppConfig
    search_results: list[SearchResult] | None = None
    session_state: ChatSessionState | None = None


@dataclass
class SearchDeps:
    """Dependencies for search agent."""

    client: HaikuRAG
    config: AppConfig
    filter: str | None = None
    search_results: list[SearchResult] = field(default_factory=list)


def build_document_filter(document_name: str) -> str:
    """Build SQL filter for document name matching."""
    escaped = document_name.replace("'", "''")
    no_spaces = escaped.replace(" ", "")
    return (
        f"LOWER(uri) LIKE LOWER('%{escaped}%') OR LOWER(title) LIKE LOWER('%{escaped}%') "
        f"OR LOWER(uri) LIKE LOWER('%{no_spaces}%') OR LOWER(title) LIKE LOWER('%{no_spaces}%')"
    )
