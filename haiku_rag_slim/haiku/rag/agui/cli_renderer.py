"""Generic CLI renderer for AG-UI events with Rich console output."""

from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console

from haiku.rag.agui.events import AGUIEvent


class AGUIConsoleRenderer:
    """Renders AG-UI events to Rich console with formatted output.

    Generic renderer that processes AG-UI protocol events and renders them
    with Rich formatting. Works with any graph that emits AG-UI events.
    """

    def __init__(self, console: Console | None = None):
        """Initialize the renderer.

        Args:
            console: Optional Rich console instance (creates new one if not provided)
        """
        self.console = console or Console()
        self._state: dict | None = None

    async def render(self, events: AsyncIterator[AGUIEvent]) -> Any | None:
        """Process events and render to console, return final result.

        Args:
            events: Async iterator of AG-UI events

        Returns:
            The final result from RunFinished event, or None
        """
        result = None

        async for event in events:
            event_type = event.get("type")

            if event_type == "RUN_STARTED":
                self._render_run_started(event)
            elif event_type == "RUN_FINISHED":
                result = event.get("result")
                self._render_run_finished()
            elif event_type == "RUN_ERROR":
                self._render_error(event)
            elif event_type == "STEP_STARTED":
                self._render_step_started(event)
            elif event_type == "STEP_FINISHED":
                self._render_step_finished(event)
            elif event_type == "TEXT_MESSAGE_CHUNK":
                self._render_text_message(event)
            elif event_type == "TEXT_MESSAGE_START":
                pass  # Start of streaming message, no output needed
            elif event_type == "TEXT_MESSAGE_CONTENT":
                self._render_text_content(event)
            elif event_type == "TEXT_MESSAGE_END":
                pass  # End of streaming message, no output needed
            elif event_type == "STATE_SNAPSHOT":
                new_state = event.get("snapshot")
                self._render_state_snapshot(new_state)
                self._state = new_state
            elif event_type == "STATE_DELTA":
                self._apply_state_delta(event)
            elif event_type == "ACTIVITY_SNAPSHOT":
                self._render_activity(event)
            elif event_type == "ACTIVITY_DELTA":
                pass  # Activity deltas don't need separate rendering

        return result

    def _render_run_started(self, _event: AGUIEvent) -> None:
        """Render run start event."""

    def _render_run_finished(self) -> None:
        """Render run completion."""

    def _render_error(self, event: AGUIEvent) -> None:
        """Render error event."""
        message = event.get("message", "Unknown error")
        self.console.print(f"[bold red][RUN_ERROR][/bold red] {message}")

    def _render_step_started(self, event: AGUIEvent) -> None:
        """Render step start event."""
        step_name = event.get("stepName", "")
        if step_name:
            display_name = step_name.replace("_", " ").title()
            self.console.print(
                f"\n[bold cyan][STEP_STARTED][/bold cyan] {display_name}"
            )

    def _render_step_finished(self, _event: AGUIEvent) -> None:
        """Render step finish event."""

    def _render_text_message(self, event: AGUIEvent) -> None:
        """Render complete text message."""
        delta = event.get("delta", "")
        self.console.print(f"[magenta][TEXT_MESSAGE][/magenta] {delta}")

    def _render_text_content(self, event: AGUIEvent) -> None:
        """Render streaming text content delta."""
        delta = event.get("delta", "")
        self.console.print(delta, end="")

    def _render_activity(self, event: AGUIEvent) -> None:
        """Render activity update."""
        content = event.get("content", "")
        if content:
            self.console.print(f"[yellow][ACTIVITY][/yellow] {content}")

    def _render_state_snapshot(self, new_state: dict | None) -> None:
        """Render state snapshot showing only what changed."""
        if not new_state:
            return

        old_state = self._state or {}
        diff = self._compute_diff(old_state, new_state)

        if not diff:
            return

        self.console.print("[blue][STATE_SNAPSHOT][/blue]")
        self.console.print(diff, style="dim")

    def _compute_diff(self, old: dict, new: dict) -> dict:
        """Compute difference between old and new state."""
        diff = {}
        for key, new_value in new.items():
            old_value = old.get(key)
            if old_value != new_value:
                if isinstance(new_value, dict) and isinstance(old_value, dict):
                    nested_diff = self._compute_diff(old_value, new_value)
                    if nested_diff:
                        diff[key] = nested_diff
                else:
                    diff[key] = new_value
        return diff

    def _apply_state_delta(self, _event: AGUIEvent) -> None:
        """Apply state delta to current state."""
