from ag_ui.core import StateDeltaEvent

from haiku.rag.agents.chat.state import (
    ChatSessionState,
    SessionContext,
)
from haiku.rag.tools.filters import (
    build_document_filter,
    build_multi_document_filter,
    combine_filters,
)
from haiku.rag.tools.qa import QAHistoryEntry


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


def test_build_multi_document_filter_empty():
    """Test build_multi_document_filter returns None for empty list."""
    result = build_multi_document_filter([])
    assert result is None


def test_build_multi_document_filter_single():
    """Test build_multi_document_filter with single document."""
    result = build_multi_document_filter(["mytest"])
    assert result is not None
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result
    # Single document should not have extra wrapping parentheses
    assert " OR (" not in result


def test_build_multi_document_filter_multiple():
    """Test build_multi_document_filter with multiple documents."""
    result = build_multi_document_filter(["doc1", "doc2"])
    assert result is not None
    # Should have OR-combined filters
    assert "doc1" in result
    assert "doc2" in result
    assert " OR (" in result


def test_combine_filters_both_none():
    """Test combine_filters with both None."""
    result = combine_filters(None, None)
    assert result is None


def test_combine_filters_first_only():
    """Test combine_filters with only first filter."""
    result = combine_filters("uri = 'test'", None)
    assert result == "uri = 'test'"


def test_combine_filters_second_only():
    """Test combine_filters with only second filter."""
    result = combine_filters(None, "title = 'doc'")
    assert result == "title = 'doc'"


def test_combine_filters_both():
    """Test combine_filters combines with AND."""
    result = combine_filters("uri = 'test'", "title = 'doc'")
    assert result == "(uri = 'test') AND (title = 'doc')"


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
    session_state = ChatSessionState(session_id="test")

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
    session_state = ChatSessionState(session_id="test")

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
    original.get_or_assign_index("chunk-a")
    original.get_or_assign_index("chunk-b")

    # Serialize
    state_dict = original.model_dump()
    assert "citation_registry" in state_dict
    assert state_dict["citation_registry"] == {"chunk-a": 1, "chunk-b": 2}

    # Deserialize (simulating AG-UI state restoration)
    restored = ChatSessionState.model_validate(state_dict)

    # Existing chunks should return their persisted indices
    assert restored.get_or_assign_index("chunk-a") == 1
    assert restored.get_or_assign_index("chunk-b") == 2
    # New chunk should get next index
    assert restored.get_or_assign_index("chunk-c") == 3


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


def test_emit_state_event_includes_session_id_when_assigned():
    """emit_state_event detects session_id change from empty to UUID.

    When session_id defaults to "" and the tool assigns a UUID,
    the delta must include session_id so clients can persist it.
    """
    from haiku.rag.agents.chat.state import emit_state_event

    current_state = ChatSessionState()  # session_id=""
    new_state = ChatSessionState(session_id="assigned-uuid-123")

    event = emit_state_event(current_state, new_state)

    assert event is not None
    session_id_op = next(
        (op for op in event.delta if op["path"] == "/session_id"), None
    )
    assert session_id_op is not None
    assert session_id_op["value"] == "assigned-uuid-123"


def test_emit_state_event_returns_none_when_no_changes():
    """emit_state_event returns None when states are identical."""
    from haiku.rag.agents.chat.state import emit_state_event

    state = ChatSessionState(session_id="test-123", qa_history=[], citations=[])

    event = emit_state_event(state, state)

    assert event is None


def test_emit_state_event_returns_delta_with_changes():
    """emit_state_event returns StateDeltaEvent with JSON Patch ops for changes."""
    from ag_ui.core import EventType, StateDeltaEvent

    from haiku.rag.agents.chat.state import emit_state_event

    current_state = ChatSessionState(session_id="test-123", qa_history=[], citations=[])
    new_state = ChatSessionState(
        session_id="test-123",
        qa_history=[QAHistoryEntry(question="Q1", answer="A1", confidence=0.9)],
        citations=[],
    )

    event = emit_state_event(current_state, new_state)

    assert isinstance(event, StateDeltaEvent)
    assert event.type == EventType.STATE_DELTA
    assert len(event.delta) > 0
    # Delta should contain an "add" operation for the new qa_history entry
    ops = event.delta
    qa_history_op = next((op for op in ops if "/qa_history" in op["path"]), None)
    assert qa_history_op is not None


def test_emit_state_event_delta_with_state_key():
    """emit_state_event wraps delta paths with state_key namespace."""
    from ag_ui.core import StateDeltaEvent

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, emit_state_event

    current_state = ChatSessionState(session_id="test-123", qa_history=[])
    new_state = ChatSessionState(
        session_id="test-123",
        qa_history=[QAHistoryEntry(question="Q1", answer="A1", confidence=0.9)],
    )

    event = emit_state_event(current_state, new_state, state_key=AGUI_STATE_KEY)

    assert isinstance(event, StateDeltaEvent)
    # Paths should be namespaced under state_key
    for op in event.delta:
        assert op["path"].startswith(f"/{AGUI_STATE_KEY}")


def test_emit_state_event_delta_produces_valid_patch():
    """emit_state_event delta can be applied to reproduce new state."""
    import jsonpatch

    from haiku.rag.agents.chat.state import emit_state_event

    current_state = ChatSessionState(
        session_id="test-123",
        qa_history=[QAHistoryEntry(question="Q1", answer="A1", confidence=0.9)],
        citations=[],
    )
    new_state = ChatSessionState(
        session_id="test-123",
        qa_history=[
            QAHistoryEntry(question="Q1", answer="A1", confidence=0.9),
            QAHistoryEntry(question="Q2", answer="A2", confidence=0.8),
        ],
        citations=[],
    )

    event = emit_state_event(current_state, new_state)
    assert isinstance(event, StateDeltaEvent)

    # Apply patch to current state and verify it produces new state
    current_snapshot = current_state.model_dump(mode="json")
    patched = jsonpatch.apply_patch(current_snapshot, event.delta)
    new_snapshot = new_state.model_dump(mode="json")
    assert patched == new_snapshot
