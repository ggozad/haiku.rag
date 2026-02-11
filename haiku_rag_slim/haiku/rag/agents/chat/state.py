from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from haiku.rag.agents.research.models import Citation

if TYPE_CHECKING:
    from haiku.rag.tools.qa import QAHistoryEntry, QASessionState
    from haiku.rag.tools.session import SessionState

AGUI_STATE_KEY = "haiku.rag.chat"


class SessionContext(BaseModel):
    """Compressed summary of conversation history for research graph."""

    summary: str = ""
    last_updated: datetime | None = None


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    initial_context: str | None = None
    citations: list[Citation] = []
    qa_history: list["QAHistoryEntry"] = []
    session_context: SessionContext | None = None
    document_filter: list[str] = []
    citation_registry: dict[str, int] = {}


def _rebuild_models(qa_history_entry_cls: type) -> None:
    """Resolve ChatSessionState forward reference to QAHistoryEntry.

    Must be called after QAHistoryEntry is defined, passing the class.
    """
    ChatSessionState.model_rebuild(
        _types_namespace={"QAHistoryEntry": qa_history_entry_cls}
    )


def build_chat_state_snapshot(
    session_state: "SessionState | None",
    qa_state: "QASessionState | None",
) -> dict[str, Any]:
    """Build a combined AG-UI chat state snapshot from current values.

    Args:
        session_state: SessionState from ToolContext.
        qa_state: QASessionState from ToolContext.

    Returns:
        Snapshot dict.
    """
    snapshot: dict[str, Any] = {}

    if session_state is not None:
        snapshot.update(
            {
                "document_filter": session_state.document_filter.copy(),
                "citation_registry": session_state.citation_registry.copy(),
                "citations": [c.model_dump() for c in session_state.citations],
            }
        )

    if qa_state is not None:
        snapshot["qa_history"] = [qa.model_dump() for qa in qa_state.qa_history]
        if qa_state.session_context:
            snapshot["session_context"] = SessionContext(
                summary=qa_state.session_context
            ).model_dump(mode="json")
        else:
            snapshot["session_context"] = None

    return snapshot
