from typing import Any

import jsonpatch
from ag_ui.core import EventType, StateDeltaEvent
from pydantic import BaseModel

from haiku.rag.agents.research.models import Citation

SESSION_NAMESPACE = "haiku.rag.session"


class SessionState(BaseModel):
    """Session-level state for AG-UI integration.

    This state is shared across toolsets and enables:
    - Dynamic document filtering
    - Stable citation indices across tool calls
    - AG-UI state synchronization
    """

    document_filter: list[str] = []
    citation_registry: dict[str, int] = {}
    citations: list[Citation] = []

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


def compute_state_delta(
    old_state: SessionState,
    new_state: SessionState,
    state_key: str | None = None,
) -> StateDeltaEvent | None:
    """Compute state delta between old and new session state.

    Returns a StateDeltaEvent if there are changes, None otherwise.
    """
    return compute_combined_state_delta(
        old_state.model_dump(mode="json"),
        new_state.model_dump(mode="json"),
        state_key=state_key,
    )


def compute_combined_state_delta(
    old_snapshot: dict[str, Any],
    new_snapshot: dict[str, Any],
    state_key: str | None = None,
) -> StateDeltaEvent | None:
    """Compute state delta between old and new combined state snapshots.

    This function computes delta for the combined chat state that includes
    both SessionState and QASessionState fields.

    Args:
        old_snapshot: Previous state dict (e.g., from ChatDeps.state format).
        new_snapshot: New state dict.
        state_key: Optional namespace key for the state (e.g., "haiku.rag.chat").

    Returns:
        StateDeltaEvent if there are changes, None otherwise.
    """
    wrapped_old = {state_key: old_snapshot} if state_key else old_snapshot
    wrapped_new = {state_key: new_snapshot} if state_key else new_snapshot

    patch = jsonpatch.make_patch(wrapped_old, wrapped_new)

    if not patch.patch:
        return None

    return StateDeltaEvent(
        type=EventType.STATE_DELTA,
        delta=patch.patch,
    )
