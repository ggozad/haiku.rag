import uuid
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent

from haiku.rag.agents.chat.context import (
    get_cached_session_context,
)
from haiku.rag.agents.chat.context import (
    trigger_background_summarization as _trigger_summarization,
)
from haiku.rag.agents.chat.prompts import CHAT_SYSTEM_PROMPT
from haiku.rag.agents.chat.state import (
    AGUI_STATE_KEY,
    ChatSessionState,
    SessionContext,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.document import create_document_toolset
from haiku.rag.tools.qa import (
    QA_SESSION_NAMESPACE,
    QASessionState,
    create_qa_toolset,
)
from haiku.rag.tools.search import create_search_toolset
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState
from haiku.rag.utils import get_model


@dataclass
class ChatDeps:
    """Dependencies for chat agent.

    Implements StateHandler protocol for AG-UI state management.
    """

    config: AppConfig
    tool_context: ToolContext
    session_id: str = ""
    state_key: str | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Get current state for AG-UI protocol.

        Combines SessionState and QASessionState into a single state dict
        matching the ChatSessionState schema expected by AG-UI clients.
        """
        snapshot: dict[str, Any] = {"session_id": self.session_id}

        # Add SessionState fields
        session_state = self.tool_context.get(SESSION_NAMESPACE, SessionState)
        if session_state is not None:
            snapshot["document_filter"] = session_state.document_filter
            snapshot["citation_registry"] = session_state.citation_registry
            snapshot["citations"] = [c.model_dump() for c in session_state.citations]

        # Add QASessionState fields
        qa_session_state = self.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
        if qa_session_state is not None:
            snapshot["qa_history"] = [
                qa.model_dump() for qa in qa_session_state.qa_history
            ]
            # Convert string to SessionContext model for frontend
            if qa_session_state.session_context:
                snapshot["session_context"] = SessionContext(
                    summary=qa_session_state.session_context
                ).model_dump(mode="json")
            else:
                snapshot["session_context"] = None

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

        # Update SessionState from incoming state
        session_state = self.tool_context.get(SESSION_NAMESPACE, SessionState)
        if session_state is not None:
            if "document_filter" in state_data:
                session_state.document_filter = state_data.get("document_filter", [])
            if "citation_registry" in state_data:
                session_state.citation_registry = state_data["citation_registry"]
            if "citations" in state_data:
                from haiku.rag.agents.research.models import Citation

                session_state.citations = [
                    Citation(**c) if isinstance(c, dict) else c
                    for c in state_data.get("citations", [])
                ]

        # Extract session_id if present, or generate one
        # Track what the client sent (for delta computation)
        incoming_session_id = state_data.get("session_id", "")

        if incoming_session_id:
            self.session_id = incoming_session_id
        elif not self.session_id:
            # Generate session_id now so ask() tool can use it
            self.session_id = str(uuid.uuid4())

        # Sync session_id to SessionState (track incoming for delta computation)
        if session_state is not None:
            session_state.session_id = self.session_id
            session_state.incoming_session_id = incoming_session_id

        # Update QASessionState from incoming state
        qa_session_state = self.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
        if qa_session_state is not None:
            if "qa_history" in state_data:
                from haiku.rag.tools.qa import QAHistoryEntry

                qa_session_state.qa_history = [
                    QAHistoryEntry(**qa) if isinstance(qa, dict) else qa
                    for qa in state_data.get("qa_history", [])
                ]

            # Track what client sent for delta computation
            incoming_session_context = state_data.get("session_context")
            if isinstance(incoming_session_context, dict):
                qa_session_state.incoming_session_context = SessionContext(
                    **incoming_session_context
                )
                qa_session_state.session_context = (
                    qa_session_state.incoming_session_context.summary
                )
            elif incoming_session_context is None:
                qa_session_state.incoming_session_context = None
                qa_session_state.session_context = None

            # Check cache for fresher session_context from background summarization
            # Cache is authoritative - always use it if available
            if self.session_id:
                cached = get_cached_session_context(self.session_id)
                if cached and cached.summary:
                    qa_session_state.session_context = cached.render_markdown()

            # Handle initial_context -> session_context for first message
            # Only applies if session_context is still empty after restoring and cache check
            if "initial_context" in state_data:
                initial = state_data.get("initial_context")
                if initial and not qa_session_state.session_context:
                    qa_session_state.session_context = initial


def create_chat_agent(
    config: AppConfig,
    client: HaikuRAG,
    context: ToolContext,
) -> Agent[ChatDeps, str]:
    """Create the chat agent with composed toolsets.

    Args:
        config: Application configuration.
        client: HaikuRAG client for database operations.
        context: ToolContext for shared state across toolsets.
            Should have SessionState and QASessionState registered
            (will be auto-registered if not present).

    Returns:
        The configured chat agent.

    Example:
        async with HaikuRAG(db_path, create=True) as client:
            context = ToolContext()
            agent = create_chat_agent(config, client, context)
            deps = ChatDeps(config=config, tool_context=context)
            result = await agent.run("Search for X", deps=deps)
    """
    # Ensure session states are registered with proper AG-UI state key
    existing = context.get(SESSION_NAMESPACE, SessionState)
    if existing is None:
        context.register(SESSION_NAMESPACE, SessionState(state_key=AGUI_STATE_KEY))
    elif existing.state_key is None:
        existing.state_key = AGUI_STATE_KEY
    if context.get(QA_SESSION_NAMESPACE, QASessionState) is None:
        context.register(QA_SESSION_NAMESPACE, QASessionState())

    # Create toolsets - these capture client, config, and context in closures
    search_toolset = create_search_toolset(client, config, context=context)
    document_toolset = create_document_toolset(client, config, context=context)
    qa_toolset = create_qa_toolset(client, config, context=context)

    # Create the agent with composed toolsets
    model = get_model(config.qa.model, config)

    agent = cast(
        Agent[ChatDeps, str],
        Agent(
            model,
            deps_type=ChatDeps,
            output_type=str,
            instructions=CHAT_SYSTEM_PROMPT,
            toolsets=[search_toolset, document_toolset, qa_toolset],  # type: ignore[arg-type]
            retries=3,
        ),
    )

    return agent


def trigger_background_summarization(deps: ChatDeps) -> None:
    """Trigger background session summarization if qa_history has entries.

    Call this after agent.run() or agent.run_stream() completes to update
    the session context summary in the background.

    Note: The ask() tool now triggers summarization internally, so this
    function is primarily for explicit triggering when needed.

    Args:
        deps: Chat dependencies with tool_context containing QASessionState.
    """
    qa_session_state = deps.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
    if qa_session_state is None or not qa_session_state.qa_history:
        return
    if not deps.session_id:
        return

    _trigger_summarization(
        qa_session_state=qa_session_state,
        config=deps.config,
        session_id=deps.session_id,
    )


async def run_chat_agent(
    agent: Agent[ChatDeps, str],
    deps: ChatDeps,
    message: str,
) -> str:
    """Run the chat agent and trigger background summarization.

    This wrapper handles post-processing like background summarization.

    Args:
        agent: The chat agent.
        deps: Chat dependencies.
        message: User message.

    Returns:
        Agent response.
    """
    result = await agent.run(message, deps=deps)
    trigger_background_summarization(deps)
    return result.output


__all__ = [
    "create_chat_agent",
    "run_chat_agent",
    "trigger_background_summarization",
    "ChatDeps",
    "ChatSessionState",
    "SessionContext",
    "AGUI_STATE_KEY",
]
