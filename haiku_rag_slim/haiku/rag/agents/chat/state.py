from datetime import datetime
from typing import TYPE_CHECKING

import jsonpatch
from ag_ui.core import EventType, StateDeltaEvent
from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation

if TYPE_CHECKING:
    from haiku.rag.tools.qa import QAHistoryEntry

AGUI_STATE_KEY = "haiku.rag.chat"


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
    initial_context: str | None = None
    citations: list[Citation] = []
    qa_history: list["QAHistoryEntry"] = []
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


def _rebuild_models(qa_history_entry_cls: type) -> None:
    """Resolve ChatSessionState forward reference to QAHistoryEntry.

    Must be called after QAHistoryEntry is defined, passing the class.
    """
    ChatSessionState.model_rebuild(
        _types_namespace={"QAHistoryEntry": qa_history_entry_cls}
    )


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
