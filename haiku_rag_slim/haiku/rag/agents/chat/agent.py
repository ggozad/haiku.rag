from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent

from haiku.rag.agents.chat.context import (
    trigger_background_summarization as _trigger_summarization,
)
from haiku.rag.agents.chat.prompts import build_chat_prompt
from haiku.rag.agents.chat.state import AGUI_STATE_KEY
from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.deps import AgentDeps
from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState
from haiku.rag.tools.session import SessionContext
from haiku.rag.tools.toolkit import (
    FEATURE_ANALYSIS,
    FEATURE_DOCUMENTS,
    FEATURE_QA,
    FEATURE_SEARCH,
    Toolkit,
    build_toolkit,
)
from haiku.rag.utils import get_model

DEFAULT_FEATURES = [FEATURE_SEARCH, FEATURE_DOCUMENTS, FEATURE_QA]


def _on_qa_complete(qa_session_state: QASessionState, config: AppConfig) -> None:
    _trigger_summarization(qa_session_state=qa_session_state, config=config)


@dataclass
class ChatDeps(AgentDeps):
    """Dependencies for chat agent.

    Extends AgentDeps with chat-specific config and state handling.
    """

    config: AppConfig = field(default_factory=AppConfig)

    @AgentDeps.state.setter
    def state(self, value: dict[str, Any] | None) -> None:
        """Set state from AG-UI protocol with chat-specific overrides."""
        if value is None:
            return

        state_data = self._extract_state_data(value)

        # Preserve server's session_context before restore overwrites it
        qa_session_state = self.tool_context.get(QA_SESSION_NAMESPACE, QASessionState)
        server_session_context = (
            qa_session_state.session_context if qa_session_state is not None else None
        )

        self.tool_context.restore_state_snapshot(state_data)

        # Chat-specific overrides after generic restore
        if qa_session_state is not None:
            # Prefer server's session_context (background summarizer may
            # have updated it since the client's last snapshot).
            if server_session_context is not None:
                qa_session_state.session_context = server_session_context

            # Handle initial_context -> session_context for first message
            if qa_session_state.session_context is None:
                if "initial_context" in state_data:
                    initial = state_data.get("initial_context")
                    if initial:
                        qa_session_state.session_context = SessionContext(
                            summary=initial
                        )


def build_chat_toolkit(
    config: AppConfig,
    features: list[str] | None = None,
) -> Toolkit:
    """Build a Toolkit configured for the chat agent.

    Includes the on_qa_complete callback that triggers background
    session summarization.

    Args:
        config: Application configuration.
        features: List of features to enable. Defaults to DEFAULT_FEATURES.

    Returns:
        A Toolkit ready for chat agent composition and context creation.
    """
    if features is None:
        features = DEFAULT_FEATURES

    return build_toolkit(config, features=features, on_qa_complete=_on_qa_complete)


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
    from haiku.rag.tools.context import prepare_context

    if features is None:
        features = DEFAULT_FEATURES

    prepare_context(context, features=features, state_key=AGUI_STATE_KEY)


def create_chat_agent(
    config: AppConfig,
    features: list[str] | None = None,
    preamble: str | None = None,
    toolkit: Toolkit | None = None,
) -> Agent[ChatDeps, str]:
    """Create the chat agent with composed toolsets.

    Args:
        config: Application configuration.
        features: List of features to enable. Defaults to DEFAULT_FEATURES
            (search, documents, qa). Available features: "search",
            "documents", "qa", "analysis".
        preamble: Optional custom identity/rules section for the system prompt.
            When provided, replaces the default identity prompt. Tool guidance,
            feature rules, and closing are still appended by the builder.
        toolkit: Optional pre-built Toolkit. When provided, its toolsets are
            used directly. When omitted, a toolkit is built from config and
            features.

    Returns:
        The configured chat agent.

    Example:
        async with HaikuRAG(db_path, create=True) as client:
            toolkit = build_chat_toolkit(config)
            context = toolkit.create_context(state_key=AGUI_STATE_KEY)
            agent = create_chat_agent(config, toolkit=toolkit)
            deps = ChatDeps(config=config, client=client, tool_context=context)
            result = await agent.run("Search for X", deps=deps)
    """
    if features is None:
        features = DEFAULT_FEATURES

    if toolkit is None:
        toolkit = build_chat_toolkit(config, features=features)

    model = get_model(config.qa.model, config)

    return Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
        instructions=build_chat_prompt(features, preamble=preamble),
        toolsets=toolkit.toolsets,
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
    """Run the chat agent.

    Args:
        agent: The chat agent.
        deps: Chat dependencies.
        message: User message.

    Returns:
        Agent response.
    """
    result = await agent.run(message, deps=deps)
    return result.output


__all__ = [
    "build_chat_toolkit",
    "create_chat_agent",
    "prepare_chat_context",
    "run_chat_agent",
    "trigger_background_summarization",
    "ChatDeps",
    "AGUI_STATE_KEY",
    "FEATURE_SEARCH",
    "FEATURE_DOCUMENTS",
    "FEATURE_QA",
    "FEATURE_ANALYSIS",
    "DEFAULT_FEATURES",
]
