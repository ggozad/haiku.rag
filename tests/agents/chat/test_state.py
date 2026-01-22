from haiku.rag.agents.chat.state import (
    MAX_QA_HISTORY,
    QAResponse,
    build_document_filter,
)


def test_build_document_filter_simple():
    """Test build_document_filter with simple name."""
    result = build_document_filter("mytest")
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result


def test_build_document_filter_with_spaces():
    """Test build_document_filter handles spaces correctly."""
    result = build_document_filter("TB MED 593")
    # Should include both the original (with spaces) and without spaces
    assert "LOWER(uri) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(uri) LIKE LOWER('%TBMED593%')" in result
    assert "LOWER(title) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(title) LIKE LOWER('%TBMED593%')" in result


def test_build_document_filter_escapes_quotes():
    """Test build_document_filter escapes single quotes."""
    result = build_document_filter("O'Reilly")
    # Single quotes should be doubled for SQL escaping
    assert "O''Reilly" in result


def test_max_qa_history_constant():
    """Test MAX_QA_HISTORY constant value."""
    assert MAX_QA_HISTORY == 50


def test_chat_session_state_background_context():
    """Test ChatSessionState accepts background_context."""
    from haiku.rag.agents.chat.state import ChatSessionState

    state = ChatSessionState(
        session_id="test-session",
        background_context="This is background knowledge about the topic.",
    )
    assert state.background_context == "This is background knowledge about the topic."


def test_chat_session_state_background_context_defaults_to_none():
    """Test ChatSessionState background_context defaults to None."""
    from haiku.rag.agents.chat.state import ChatSessionState

    state = ChatSessionState(session_id="test-session")
    assert state.background_context is None


def test_chat_deps_state_getter_returns_namespaced_state():
    """Test ChatDeps.state getter returns state under namespaced key."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(
        session_id="test-123",
        qa_history=[
            QAResponse(question="Q1", answer="A1", confidence=0.9),
        ],
        background_context="Background info",
    )

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    state = deps.state
    assert state is not None
    assert AGUI_STATE_KEY in state
    assert state[AGUI_STATE_KEY]["session_id"] == "test-123"
    assert len(state[AGUI_STATE_KEY]["qa_history"]) == 1
    assert state[AGUI_STATE_KEY]["qa_history"][0]["question"] == "Q1"
    assert state[AGUI_STATE_KEY]["background_context"] == "Background info"


def test_chat_deps_state_getter_without_namespace():
    """Test ChatDeps.state getter returns flat state when no state_key."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="test-123")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=None,
    )

    state = deps.state
    assert state is not None
    assert "session_id" in state
    assert state["session_id"] == "test-123"


def test_chat_deps_state_getter_returns_none_without_session():
    """Test ChatDeps.state getter returns None when no session_state."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps

    mock_client = MagicMock()
    mock_config = MagicMock()

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=None,
    )

    assert deps.state is None


def test_chat_deps_state_setter_updates_from_namespaced_state():
    """Test ChatDeps.state setter updates session_state from namespaced incoming state."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="initial")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    # Simulate incoming AG-UI state with namespaced key
    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "updated-123",
            "qa_history": [
                {"question": "Q1", "answer": "A1", "confidence": 0.9, "citations": []}
            ],
            "citations": [],
            "background_context": "New context",
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert deps.session_state.session_id == "updated-123"
    assert len(deps.session_state.qa_history) == 1
    assert deps.session_state.qa_history[0].question == "Q1"
    assert deps.session_state.background_context == "New context"


def test_chat_deps_state_setter_handles_none():
    """Test ChatDeps.state setter handles None gracefully."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="original")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
    )

    # Setting None should not raise and should not change state
    deps.state = None

    assert deps.session_state is not None
    assert deps.session_state.session_id == "original"


def test_chat_deps_state_setter_without_session_state():
    """Test ChatDeps.state setter does nothing when session_state is None."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps

    mock_client = MagicMock()
    mock_config = MagicMock()

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=None,
    )

    # Should not raise even with valid incoming state
    deps.state = {"session_id": "test", "qa_history": [], "citations": []}

    assert deps.session_state is None


def test_chat_deps_state_setter_with_citation_dicts():
    """Test ChatDeps.state setter converts citation dicts to Citation."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="test")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "test",
            "qa_history": [],
            "citations": [
                {
                    "index": 1,
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "document_uri": "test.md",
                    "document_title": "Test Doc",
                    "page_numbers": [1, 2],
                    "headings": ["Intro"],
                    "content": "Test content",
                }
            ],
            "background_context": None,
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert len(deps.session_state.citations) == 1
    citation = deps.session_state.citations[0]
    assert citation.document_id == "doc-1"
    assert citation.chunk_id == "chunk-1"
    assert citation.page_numbers == [1, 2]


def test_chat_deps_state_getter_includes_session_context():
    """Test ChatDeps.state getter includes session_context when present."""
    from datetime import datetime
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import (
        AGUI_STATE_KEY,
        ChatDeps,
        ChatSessionState,
        SessionContext,
    )

    mock_client = MagicMock()
    mock_config = MagicMock()

    now = datetime.now()
    session_state = ChatSessionState(
        session_id="test-123",
        session_context=SessionContext(
            summary="User discussed authentication.",
            last_updated=now,
        ),
    )

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    state = deps.state
    assert state is not None
    assert AGUI_STATE_KEY in state
    assert state[AGUI_STATE_KEY]["session_context"] is not None
    assert (
        state[AGUI_STATE_KEY]["session_context"]["summary"]
        == "User discussed authentication."
    )


def test_chat_deps_state_setter_restores_session_context():
    """Test ChatDeps.state setter restores session_context from incoming state."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="initial")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "test-123",
            "qa_history": [],
            "citations": [],
            "background_context": None,
            "session_context": {
                "summary": "Restored context summary.",
                "last_updated": "2025-01-15T10:30:00",
            },
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert deps.session_state.session_context is not None
    assert deps.session_state.session_context.summary == "Restored context summary."


def test_chat_deps_state_setter_handles_null_session_context():
    """Test ChatDeps.state setter handles null session_context."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import (
        AGUI_STATE_KEY,
        ChatDeps,
        ChatSessionState,
        SessionContext,
    )

    mock_client = MagicMock()
    mock_config = MagicMock()

    # Start with a session_context
    session_state = ChatSessionState(
        session_id="test",
        session_context=SessionContext(summary="Initial summary"),
    )
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    # Send null to clear it
    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "test",
            "qa_history": [],
            "citations": [],
            "session_context": None,
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert deps.session_state.session_context is None
