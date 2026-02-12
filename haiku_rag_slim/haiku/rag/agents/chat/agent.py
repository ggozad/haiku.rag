from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from haiku.rag.agents.chat.context import (
    trigger_background_summarization as _trigger_summarization,
)
from haiku.rag.agents.chat.prompts import build_chat_prompt
from haiku.rag.agents.chat.state import (
    AGUI_STATE_KEY,
    ChatSessionState,
    build_chat_state_snapshot,
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
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionContext, SessionState
from haiku.rag.utils import get_model

FEATURE_SEARCH = "search"
FEATURE_DOCUMENTS = "documents"
FEATURE_QA = "qa"
FEATURE_ANALYSIS = "analysis"

DEFAULT_FEATURES = [FEATURE_SEARCH, FEATURE_DOCUMENTS, FEATURE_QA]


@dataclass
class ChatDeps:
    """Dependencies for chat agent.

    Implements RAGDeps protocol and StateHandler protocol for AG-UI state management.
    """

    config: AppConfig
    client: HaikuRAG
    tool_context: ToolContext
    state_key: str | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Get current state for AG-UI protocol.

        Combines SessionState and QASessionState into a single state dict
        matching the ChatSessionState schema expected by AG-UI clients.
        """
        session_state = self.tool_context.get(SESSION_NAMESPACE, SessionState)
        qa_session_state = self.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
        snapshot = build_chat_state_snapshot(session_state, qa_session_state)
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

        qa_session_state = self.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
        if qa_session_state is not None:
            if "qa_history" in state_data:
                from haiku.rag.tools.qa import QAHistoryEntry

                qa_session_state.qa_history = [
                    QAHistoryEntry(**qa) if isinstance(qa, dict) else qa
                    for qa in state_data.get("qa_history", [])
                ]

            # Prefer server's session_context (background summarizer may
            # have updated it since the client's last snapshot).
            if qa_session_state.session_context is None:
                session_context = state_data.get("session_context")
                if isinstance(session_context, dict):
                    qa_session_state.session_context = SessionContext(**session_context)

                # Handle initial_context -> session_context for first message
                if "initial_context" in state_data:
                    initial = state_data.get("initial_context")
                    if initial and qa_session_state.session_context is None:
                        qa_session_state.session_context = SessionContext(
                            summary=initial
                        )


def prepare_chat_context(
    context: ToolContext,
    features: list[str] | None = None,
) -> None:
    """Register required namespaces in a ToolContext for chat agent use.

    Idempotent — safe to call multiple times on the same context.

    Args:
        context: ToolContext to prepare.
        features: List of enabled features. Defaults to DEFAULT_FEATURES.
    """
    if features is None:
        features = DEFAULT_FEATURES

    if context.get(SESSION_NAMESPACE, SessionState) is None:
        context.register(SESSION_NAMESPACE, SessionState())
    if context.state_key is None:
        context.state_key = AGUI_STATE_KEY

    if FEATURE_QA in features:
        if context.get(QA_SESSION_NAMESPACE, QASessionState) is None:
            context.register(QA_SESSION_NAMESPACE, QASessionState())


def create_chat_agent(
    config: AppConfig,
    features: list[str] | None = None,
) -> Agent[ChatDeps, str]:
    """Create the chat agent with composed toolsets.

    Args:
        config: Application configuration.
        features: List of features to enable. Defaults to DEFAULT_FEATURES
            (search, documents, qa). Available features: "search",
            "documents", "qa", "analysis".

    Returns:
        The configured chat agent.

    Example:
        async with HaikuRAG(db_path, create=True) as client:
            context = ToolContext()
            prepare_chat_context(context)
            agent = create_chat_agent(config)
            deps = ChatDeps(config=config, client=client, tool_context=context)
            result = await agent.run("Search for X", deps=deps)
    """
    if features is None:
        features = DEFAULT_FEATURES

    toolsets = []
    if FEATURE_SEARCH in features:
        toolsets.append(create_search_toolset(config))
    if FEATURE_DOCUMENTS in features:
        toolsets.append(create_document_toolset(config))
    if FEATURE_QA in features:
        toolsets.append(create_qa_toolset(config))
    if FEATURE_ANALYSIS in features:
        from haiku.rag.tools.analysis import create_analysis_toolset

        toolsets.append(create_analysis_toolset(config))

    model = get_model(config.qa.model, config)

    return Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
        instructions=build_chat_prompt(features),
        toolsets=toolsets,
        retries=3,
    )


def trigger_background_summarization(deps: ChatDeps) -> None:
    """Trigger background session summarization if qa_history has entries.

    Call this after agent.run() or agent.run_stream() completes to update
    the session context summary in the background.

    Args:
        deps: Chat dependencies with tool_context containing QASessionState.
    """
    qa_session_state = deps.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
    if qa_session_state is None or not qa_session_state.qa_history:
        return

    _trigger_summarization(
        qa_session_state=qa_session_state,
        config=deps.config,
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
    "prepare_chat_context",
    "run_chat_agent",
    "trigger_background_summarization",
    "ChatDeps",
    "ChatSessionState",
    "AGUI_STATE_KEY",
    "FEATURE_SEARCH",
    "FEATURE_DOCUMENTS",
    "FEATURE_QA",
    "FEATURE_ANALYSIS",
    "DEFAULT_FEATURES",
]
