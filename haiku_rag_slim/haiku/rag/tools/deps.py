from dataclasses import dataclass
from typing import Any

from haiku.rag.client import HaikuRAG
from haiku.rag.tools.context import ToolContext


@dataclass
class AgentDeps:
    """Generic dependencies for agents using haiku.rag toolsets.

    Implements RAGDeps protocol and AG-UI state protocol.
    """

    client: HaikuRAG
    tool_context: ToolContext
    state_key: str | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Get current state for AG-UI protocol."""
        snapshot = self.tool_context.build_state_snapshot()
        if self.state_key:
            return {self.state_key: snapshot}
        return snapshot

    @state.setter
    def state(self, value: dict[str, Any] | None) -> None:
        """Set state from AG-UI protocol."""
        if value is None:
            return

        data: dict[str, Any] = value
        if self.state_key and self.state_key in value:
            nested = value[self.state_key]
            if isinstance(nested, dict):
                data = nested
        self.tool_context.restore_state_snapshot(data)
