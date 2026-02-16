from typing import TYPE_CHECKING

from pydantic import BaseModel

from haiku.rag.agents.research.models import Citation
from haiku.rag.tools.session import SessionContext

if TYPE_CHECKING:
    from haiku.rag.tools.qa import QAHistoryEntry

AGUI_STATE_KEY = "haiku.rag.chat"


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    initial_context: str | None = None
    citations: list[Citation] = []
    citations_history: list[list[Citation]] = []
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
