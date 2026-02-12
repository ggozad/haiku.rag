from unittest.mock import MagicMock

import pytest

from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.deps import AgentDeps
from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState


@pytest.fixture
def mock_client():
    return MagicMock()


def test_agent_deps_state_getter_empty(mock_client):
    """state returns empty dict when no namespaces are registered."""
    ctx = ToolContext()
    deps = AgentDeps(client=mock_client, tool_context=ctx)
    assert deps.state == {}


def test_agent_deps_state_getter_with_session(mock_client):
    """state returns flat snapshot of registered namespaces."""
    ctx = ToolContext()
    ctx.register(SESSION_NAMESPACE, SessionState())
    deps = AgentDeps(client=mock_client, tool_context=ctx)
    snapshot = deps.state
    assert "citations" in snapshot
    assert "citation_registry" in snapshot


def test_agent_deps_state_getter_with_state_key(mock_client):
    """state wraps snapshot under state_key when set."""
    ctx = ToolContext()
    ctx.register(SESSION_NAMESPACE, SessionState())
    deps = AgentDeps(client=mock_client, tool_context=ctx, state_key="my_app")
    snapshot = deps.state
    assert "my_app" in snapshot
    assert "citations" in snapshot["my_app"]


def test_agent_deps_state_setter_restores(mock_client):
    """state setter restores namespace fields from flat dict."""
    ctx = ToolContext()
    ctx.register(SESSION_NAMESPACE, SessionState())
    deps = AgentDeps(client=mock_client, tool_context=ctx)
    deps.state = {"document_filter": ["doc1", "doc2"]}
    session = ctx.get(SESSION_NAMESPACE, SessionState)
    assert session is not None
    assert session.document_filter == ["doc1", "doc2"]


def test_agent_deps_state_setter_with_state_key(mock_client):
    """state setter extracts data from namespaced key."""
    ctx = ToolContext()
    ctx.register(SESSION_NAMESPACE, SessionState())
    deps = AgentDeps(client=mock_client, tool_context=ctx, state_key="my_app")
    deps.state = {"my_app": {"document_filter": ["doc1"]}}
    session = ctx.get(SESSION_NAMESPACE, SessionState)
    assert session is not None
    assert session.document_filter == ["doc1"]


def test_agent_deps_state_setter_ignores_none(mock_client):
    """state setter is a no-op when value is None."""
    ctx = ToolContext()
    ctx.register(SESSION_NAMESPACE, SessionState())
    deps = AgentDeps(client=mock_client, tool_context=ctx)
    deps.state = None
    session = ctx.get(SESSION_NAMESPACE, SessionState)
    assert session is not None
    assert session.document_filter == []


def test_agent_deps_state_roundtrip(mock_client):
    """Build snapshot then restore produces equivalent state."""
    ctx = ToolContext()
    ctx.register(SESSION_NAMESPACE, SessionState(document_filter=["doc1"]))
    ctx.register(QA_SESSION_NAMESPACE, QASessionState())
    deps = AgentDeps(client=mock_client, tool_context=ctx, state_key="app")

    snapshot = deps.state

    ctx2 = ToolContext()
    ctx2.register(SESSION_NAMESPACE, SessionState())
    ctx2.register(QA_SESSION_NAMESPACE, QASessionState())
    deps2 = AgentDeps(client=mock_client, tool_context=ctx2, state_key="app")
    deps2.state = snapshot

    session = ctx2.get(SESSION_NAMESPACE, SessionState)
    assert session is not None
    assert session.document_filter == ["doc1"]


def test_agent_deps_satisfies_rag_deps_protocol(mock_client):
    """AgentDeps satisfies the RAGDeps protocol."""
    from haiku.rag.tools.context import RAGDeps

    ctx = ToolContext()
    deps = AgentDeps(client=mock_client, tool_context=ctx)
    assert isinstance(deps, RAGDeps)
