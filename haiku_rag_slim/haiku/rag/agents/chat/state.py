from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from haiku.rag.agents.research.models import Citation
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

    @property
    def sources(self) -> list[str]:
        """Source names for display."""
        return list(
            dict.fromkeys(c.document_title or c.document_uri for c in self.citations)
        )


class SessionContext(BaseModel):
    """Compressed summary of conversation history for research graph."""

    summary: str = ""
    last_updated: datetime | None = None

    def render_markdown(self) -> str:
        """Render context for injection into research graph."""
        return self.summary


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    session_id: str = ""
    citations: list[Citation] = []
    qa_history: list[QAResponse] = []
    session_context: SessionContext | None = None


@dataclass
class ChatDeps:
    """Dependencies for chat agent.

    Implements StateHandler protocol for AG-UI state management.
    """

    client: HaikuRAG
    config: AppConfig
    search_results: list[SearchResult] | None = None
    session_state: ChatSessionState | None = None
    state_key: str | None = None

    @property
    def state(self) -> dict[str, Any] | None:
        """Get current state for AG-UI protocol."""
        if self.session_state is None:
            return None
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
        if self.session_state is not None:
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
            # NOTE: session_context intentionally NOT updated from client
            # The agent owns this via server-side cache


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
