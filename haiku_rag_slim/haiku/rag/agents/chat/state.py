import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import jsonpatch
from ag_ui.core import EventType, StateDeltaEvent
from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation, SearchAnswer
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult

MAX_QA_HISTORY = 50

AGUI_STATE_KEY = "haiku.rag.chat"


class QAResponse(BaseModel):
    """A Q&A pair from conversation history with citations."""

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


class DocumentInfo(BaseModel):
    """Document info for list_documents response."""

    title: str
    uri: str
    created: str


class DocumentListResponse(BaseModel):
    """Response from list_documents tool."""

    documents: list[DocumentInfo]
    page: int
    total_pages: int
    total_documents: int


class SessionContext(BaseModel):
    """Compressed summary of conversation history for research graph."""

    summary: str = ""
    last_updated: datetime | None = None

    def render_markdown(self) -> str:
        """Render context for injection into research graph."""
        return self.summary


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    initial_context: str | None = None
    citations: list[Citation] = []
    qa_history: list[QAResponse] = []
    session_context: SessionContext | None = None
    document_filter: list[str] = []
    citation_registry: dict[str, int] = {}

    def get_or_assign_index(self, chunk_id: str) -> int:
        """Get or assign a stable citation index for a chunk_id.

        Citation indices persist across tool calls within a session.
        The first chunk gets index 1, subsequent new chunks get incrementing indices.
        Same chunk_id always returns the same index.
        """
        if chunk_id in self.citation_registry:
            return self.citation_registry[chunk_id]

        new_index = len(self.citation_registry) + 1
        self.citation_registry[chunk_id] = new_index
        return new_index


@dataclass
class ChatDeps:
    """Dependencies for chat agent.

    Implements StateHandler protocol for AG-UI state management.
    """

    client: HaikuRAG
    config: AppConfig
    search_results: list[SearchResult] | None = None
    session_state: ChatSessionState = field(
        default_factory=lambda: ChatSessionState(session_id="")
    )
    state_key: str | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Get current state for AG-UI protocol."""
        snapshot = self.session_state.model_dump()
        if self.state_key:
            return {self.state_key: snapshot}
        return snapshot

    @state.setter
    def state(self, value: dict[str, Any] | None) -> None:
        """Set state from AG-UI protocol."""
        if value is None:
            return
        # Extract from namespaced key if present
        state_data: dict[str, Any] = value
        if self.state_key and self.state_key in value:
            nested = value[self.state_key]
            if isinstance(nested, dict):
                state_data = nested
        # Update session_state from incoming state
        if "qa_history" in state_data:
            self.session_state.qa_history = [
                QAResponse(**qa) if isinstance(qa, dict) else qa
                for qa in state_data.get("qa_history", [])
            ]
        if "citations" in state_data:
            self.session_state.citations = [
                Citation(**c) if isinstance(c, dict) else c
                for c in state_data.get("citations", [])
            ]
        if state_data.get("session_id"):
            self.session_state.session_id = state_data["session_id"]
        if "document_filter" in state_data:
            self.session_state.document_filter = state_data.get("document_filter", [])
        if "citation_registry" in state_data:
            self.session_state.citation_registry = state_data["citation_registry"]
        if "initial_context" in state_data:
            self.session_state.initial_context = state_data.get("initial_context")


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


def build_multi_document_filter(document_names: list[str]) -> str | None:
    """Build SQL filter for multiple document names (OR combined)."""
    if not document_names:
        return None
    filters = [build_document_filter(name) for name in document_names]
    if len(filters) == 1:
        return filters[0]
    return " OR ".join(f"({f})" for f in filters)


def combine_filters(filter1: str | None, filter2: str | None) -> str | None:
    """Combine two SQL filters with AND logic."""
    filters = [f for f in [filter1, filter2] if f]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return f"({filters[0]}) AND ({filters[1]})"


def emit_state_event(
    current_state: ChatSessionState,
    new_state: ChatSessionState,
    state_key: str | None = None,
) -> StateDeltaEvent | None:
    """Emit state delta against current state, or None if no changes."""
    new_snapshot = new_state.model_dump(mode="json")
    wrapped_new = {state_key: new_snapshot} if state_key else new_snapshot

    current_snapshot = current_state.model_dump(mode="json")
    wrapped_current = {state_key: current_snapshot} if state_key else current_snapshot

    patch = jsonpatch.make_patch(wrapped_current, wrapped_new)

    if not patch.patch:
        return None

    return StateDeltaEvent(
        type=EventType.STATE_DELTA,
        delta=patch.patch,
    )
