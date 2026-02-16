import pytest

from haiku.rag.config import Config
from haiku.rag.tools.prompts import build_tools_prompt
from haiku.rag.tools.toolkit import (
    FEATURE_ANALYSIS,
    FEATURE_DOCUMENTS,
    FEATURE_QA,
    FEATURE_SEARCH,
    build_toolkit,
)


def test_build_toolkit_default_features():
    """Defaults to ["search", "documents"], producing 2 toolsets."""
    toolkit = build_toolkit(Config)
    assert len(toolkit.toolsets) == 2
    assert toolkit.features == [FEATURE_SEARCH, FEATURE_DOCUMENTS]


def test_build_toolkit_all_features():
    """All 4 features produce 4 toolsets."""
    features = [FEATURE_SEARCH, FEATURE_DOCUMENTS, FEATURE_QA, FEATURE_ANALYSIS]
    toolkit = build_toolkit(Config, features=features)
    assert len(toolkit.toolsets) == 4
    assert toolkit.features == features


def test_build_toolkit_single_feature():
    """Single feature produces 1 toolset."""
    toolkit = build_toolkit(Config, features=[FEATURE_SEARCH])
    assert len(toolkit.toolsets) == 1
    assert toolkit.features == [FEATURE_SEARCH]


def test_build_toolkit_prompt_matches_features():
    """Toolkit prompt matches build_tools_prompt for the same features."""
    features = [FEATURE_SEARCH, FEATURE_DOCUMENTS, FEATURE_QA]
    toolkit = build_toolkit(Config, features=features)
    expected = build_tools_prompt(features)
    assert toolkit.prompt == expected


def test_toolkit_create_context_registers_namespaces():
    """create_context registers correct namespaces for the features."""
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState
    from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState

    toolkit = build_toolkit(Config, features=[FEATURE_SEARCH, FEATURE_QA])
    context = toolkit.create_context()

    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is not None


def test_toolkit_create_context_no_qa_skips_qa_state():
    """create_context without QA feature skips QASessionState."""
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState
    from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState

    toolkit = build_toolkit(Config, features=[FEATURE_SEARCH, FEATURE_DOCUMENTS])
    context = toolkit.create_context()

    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.get(QA_SESSION_NAMESPACE, QASessionState) is None


def test_toolkit_create_context_sets_state_key():
    """create_context propagates state_key to the ToolContext."""
    toolkit = build_toolkit(Config)
    context = toolkit.create_context(state_key="my.state.key")
    assert context.state_key == "my.state.key"


def test_toolkit_create_context_no_state_key():
    """create_context without state_key leaves it None."""
    toolkit = build_toolkit(Config)
    context = toolkit.create_context()
    assert context.state_key is None


def test_toolkit_prepare_existing_context():
    """prepare registers namespaces on an existing ToolContext."""
    from haiku.rag.tools.context import ToolContext
    from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState

    toolkit = build_toolkit(Config, features=[FEATURE_SEARCH])
    context = ToolContext()
    toolkit.prepare(context, state_key="test.key")

    assert context.get(SESSION_NAMESPACE, SessionState) is not None
    assert context.state_key == "test.key"


def test_toolkit_frozen():
    """Toolkit is immutable after creation."""
    toolkit = build_toolkit(Config)
    with pytest.raises(AttributeError):
        toolkit.features = ["search"]  # type: ignore[misc]
