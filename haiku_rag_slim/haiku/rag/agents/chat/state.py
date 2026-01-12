from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import format_as_xml

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult


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
