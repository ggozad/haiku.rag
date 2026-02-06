from haiku.rag.agents.chat.state import (
    ChatSessionState,
    SessionContext,
)
from haiku.rag.tools.session import SessionState


def test_max_qa_history_constant():
    """Test MAX_QA_HISTORY constant value."""
    from haiku.rag.tools.qa import MAX_QA_HISTORY

    assert MAX_QA_HISTORY == 50


def test_citation_registry_index_assignment():
    """Test get_or_assign_index basic index assignment behavior.

    Verifies:
    - First chunk gets index 1
    - Second unique chunk gets index 2
    - Same chunk_id always returns same index
    """
    session_state = SessionState()

    # First chunk gets index 1
    index1 = session_state.get_or_assign_index("chunk-abc")
    assert index1 == 1

    # Second unique chunk gets index 2
    index2 = session_state.get_or_assign_index("chunk-def")
    assert index2 == 2

    # Same chunk_id returns same index (not incremented)
    index1_again = session_state.get_or_assign_index("chunk-abc")
    assert index1_again == 1


def test_citation_registry_stability():
    """Test citation indices are stable across multiple calls in any order."""
    session_state = SessionState()

    # First round assigns indices 1, 2, 3
    idx_a = session_state.get_or_assign_index("chunk-a")
    idx_b = session_state.get_or_assign_index("chunk-b")
    idx_c = session_state.get_or_assign_index("chunk-c")

    # Second round - existing chunks keep their indices regardless of order
    assert session_state.get_or_assign_index("chunk-b") == idx_b
    assert session_state.get_or_assign_index("chunk-a") == idx_a
    assert session_state.get_or_assign_index("chunk-c") == idx_c

    # New chunk gets next index
    idx_d = session_state.get_or_assign_index("chunk-d")
    assert idx_d == 4


def test_citation_registry_serialization_roundtrip():
    """Test citation_registry serializes and deserializes correctly for AG-UI state."""
    # Create state and assign indices
    original = ChatSessionState(session_id="test")
    original.citation_registry = {"chunk-a": 1, "chunk-b": 2}

    # Serialize
    state_dict = original.model_dump()
    assert "citation_registry" in state_dict
    assert state_dict["citation_registry"] == {"chunk-a": 1, "chunk-b": 2}

    # Deserialize (simulating AG-UI state restoration)
    restored = ChatSessionState.model_validate(state_dict)
    assert restored.citation_registry == {"chunk-a": 1, "chunk-b": 2}


def test_chat_session_state_defaults_to_empty_session_id():
    """New ChatSessionState should default to empty session_id.

    Tools in agent.py detect the empty string and assign a UUID,
    which then appears in the state delta so clients receive it.
    """
    state = ChatSessionState()
    assert state.session_id == ""


def test_chat_session_state_preserves_explicit_session_id():
    """Explicit session_id should be preserved."""
    state = ChatSessionState(session_id="my-custom-id")
    assert state.session_id == "my-custom-id"


def test_chat_session_state_initial_context_default_none():
    """Initial context should default to None."""
    state = ChatSessionState()
    assert state.initial_context is None


def test_chat_session_state_initial_context_preserved():
    """Explicit initial_context should be preserved."""
    state = ChatSessionState(initial_context="Background info about the project")
    assert state.initial_context == "Background info about the project"


def test_chat_session_state_initial_context_serialization():
    """initial_context should serialize and deserialize correctly."""
    state = ChatSessionState(
        session_id="test-123",
        initial_context="User is working on authentication",
    )
    state_dict = state.model_dump()
    assert state_dict["initial_context"] == "User is working on authentication"

    restored = ChatSessionState.model_validate(state_dict)
    assert restored.initial_context == "User is working on authentication"


def test_chat_session_state_model_dump_json_serializes_datetime():
    """model_dump(mode='json') should serialize datetime to ISO string.

    Agent tools use model_dump(mode='json') when creating StateSnapshotEvent
    to ensure datetime fields are JSON-serializable for external clients
    persisting AG-UI state to database JSON columns.
    """
    from datetime import datetime

    session_state = ChatSessionState(
        session_id="test",
        session_context=SessionContext(
            summary="Test summary",
            last_updated=datetime(2025, 1, 27, 12, 0, 0),
        ),
    )

    # This is how agent.py creates snapshots for StateSnapshotEvent
    snapshot = session_state.model_dump(mode="json")

    # datetime should be serialized as ISO string, not datetime object
    assert isinstance(snapshot["session_context"]["last_updated"], str)
    assert snapshot["session_context"]["last_updated"] == "2025-01-27T12:00:00"
