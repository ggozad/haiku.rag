from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation, SearchAnswer

PRIOR_ANSWER_RELEVANCE_THRESHOLD = 0.7


class QAHistoryEntry(BaseModel):
    """A Q&A pair with optional cached embedding for similarity matching."""

    question: str
    answer: str
    confidence: float = 0.9
    citations: list[Citation] = []
    question_embedding: list[float] | None = Field(default=None, exclude=True)

    @property
    def sources(self) -> list[str]:
        """Source names for display."""
        return list(
            dict.fromkeys(c.document_title or c.document_uri for c in self.citations)
        )

    def to_search_answer(self) -> SearchAnswer:
        """Convert to SearchAnswer for research graph context."""
        return SearchAnswer(
            query=self.question,
            answer=self.answer,
            confidence=self.confidence,
            cited_chunks=[c.chunk_id for c in self.citations],
            citations=self.citations,
        )
