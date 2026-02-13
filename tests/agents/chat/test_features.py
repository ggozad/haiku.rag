from pydantic_ai import FunctionToolset

from haiku.rag.agents.chat.agent import (
    DEFAULT_FEATURES,
    FEATURE_ANALYSIS,
    FEATURE_DOCUMENTS,
    FEATURE_QA,
    FEATURE_SEARCH,
    ChatDeps,
    create_chat_agent,
    prepare_chat_context,
)
from haiku.rag.agents.chat.prompts import build_chat_prompt
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState


def _count_function_toolsets(agent) -> int:
    """Count FunctionToolset instances in an agent (excludes internal toolsets)."""
    return sum(1 for t in agent.toolsets if type(t) is FunctionToolset)


# =============================================================================
# Feature Selection Tests
# =============================================================================


def test_default_features(temp_db_path):
    """Default features create search + document + qa toolsets and register both states."""
    context = ToolContext()
    prepare_chat_context(context)
    agent = create_chat_agent(Config)

    # Should have 3 toolsets (search, document, qa)
    assert _count_function_toolsets(agent) == 3

    # Both SessionState and QASessionState should be registered
    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is not None


def test_search_only(temp_db_path):
    """features=["search"] creates only search toolset, no QASessionState."""
    context = ToolContext()
    prepare_chat_context(context, features=[FEATURE_SEARCH])
    agent = create_chat_agent(Config, features=[FEATURE_SEARCH])

    assert _count_function_toolsets(agent) == 1

    # SessionState always registered, but QASessionState should NOT be
    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is None


def test_search_and_documents(temp_db_path):
    """features=["search", "documents"] creates both toolsets, no QASessionState."""
    context = ToolContext()
    prepare_chat_context(context, features=[FEATURE_SEARCH, FEATURE_DOCUMENTS])
    agent = create_chat_agent(Config, features=[FEATURE_SEARCH, FEATURE_DOCUMENTS])

    assert _count_function_toolsets(agent) == 2

    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is None


def test_all_features(temp_db_path):
    """All four features create four toolsets."""
    context = ToolContext()
    prepare_chat_context(
        context,
        features=[FEATURE_SEARCH, FEATURE_DOCUMENTS, FEATURE_QA, FEATURE_ANALYSIS],
    )
    agent = create_chat_agent(
        Config,
        features=[FEATURE_SEARCH, FEATURE_DOCUMENTS, FEATURE_QA, FEATURE_ANALYSIS],
    )

    assert _count_function_toolsets(agent) == 4

    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is not None


def test_no_qa_skips_qa_session_state(temp_db_path):
    """Without QA feature, QASessionState is not registered."""
    context = ToolContext()
    prepare_chat_context(context, features=[FEATURE_SEARCH, FEATURE_DOCUMENTS])

    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is None


def test_chat_deps_state_without_qa(temp_db_path):
    """ChatDeps.state getter omits qa_history/session_context when QASessionState absent."""
    from haiku.rag.agents.chat.state import AGUI_STATE_KEY

    client = HaikuRAG(temp_db_path, create=True)
    context = ToolContext()
    prepare_chat_context(context, features=[FEATURE_SEARCH])

    deps = ChatDeps(config=Config, client=client, tool_context=context)
    state = deps.state

    # State is wrapped under the AGUI state key
    assert AGUI_STATE_KEY in state
    inner = state[AGUI_STATE_KEY]

    # SessionState fields should be present
    assert "document_filter" in inner
    assert "citation_registry" in inner
    assert "citations" in inner
    # QA fields should NOT be present
    assert "qa_history" not in inner
    assert "session_context" not in inner
    client.close()


# =============================================================================
# Prompt Composition Tests
# =============================================================================


def test_build_chat_prompt_default():
    """Default features produce prompt mentioning all standard tools."""
    prompt = build_chat_prompt(DEFAULT_FEATURES)

    assert "list_documents" in prompt
    assert "get_document" in prompt
    assert "summarize_document" in prompt
    assert "ask" in prompt
    assert "search" in prompt
    assert "analyze" not in prompt


def test_build_chat_prompt_search_only():
    """Search-only prompt doesn't mention ask or document tools."""
    prompt = build_chat_prompt([FEATURE_SEARCH])

    assert "search" in prompt
    assert '"ask"' not in prompt
    assert '"list_documents"' not in prompt
    assert '"get_document"' not in prompt
    assert '"summarize_document"' not in prompt


def test_build_chat_prompt_includes_analysis():
    """Analysis feature adds analyze guidance to prompt."""
    prompt = build_chat_prompt(
        [FEATURE_SEARCH, FEATURE_QA, FEATURE_DOCUMENTS, FEATURE_ANALYSIS]
    )

    assert "analyze" in prompt
    assert "search" in prompt
    assert "ask" in prompt
