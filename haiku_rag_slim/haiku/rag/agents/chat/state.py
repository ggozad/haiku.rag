from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from haiku.rag.agents.research.models import Citation

if TYPE_CHECKING:
    from haiku.rag.tools.qa import QAHistoryEntry

AGUI_STATE_KEY = "haiku.rag.chat"


class SessionContext(BaseModel):
    """Compressed summary of conversation history for research graph."""

    summary: str = ""
    last_updated: datetime | None = None


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    session_id: str = ""
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
