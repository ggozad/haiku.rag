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

    @property
    def state(self) -> dict[str, Any]:
        """Get current state for AG-UI protocol."""
        snapshot = self.tool_context.build_state_snapshot()
        state_key = self.tool_context.state_key
        if state_key:
            return {state_key: snapshot}
        return snapshot

    @state.setter
    def state(self, value: dict[str, Any] | None) -> None:
        """Set state from AG-UI protocol."""
        if value is None:
            return
        data = self._extract_state_data(value)
        self.tool_context.restore_state_snapshot(data)

    def _extract_state_data(self, value: dict[str, Any]) -> dict[str, Any]:
        """Extract flat state dict, unwrapping state_key if present."""
        state_key = self.tool_context.state_key
        if state_key and state_key in value:
            nested = value[state_key]
            if isinstance(nested, dict):
                return nested
        return value
