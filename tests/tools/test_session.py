from ag_ui.core import EventType, StateDeltaEvent

from haiku.rag.agents.research.models import Citation
from haiku.rag.tools.session import (
    SessionState,
    compute_combined_state_delta,
    compute_state_delta,
)


class TestSessionState:
    """Tests for SessionState model."""

    def test_citations_history_defaults_to_empty(self):
        """SessionState.citations_history defaults to empty list."""
        state = SessionState()
        assert state.citations_history == []

    def test_citations_history_serialization_roundtrip(self):
        """citations_history survives serialize/deserialize."""
        citation = Citation(
            index=1,
            document_id="d1",
            chunk_id="c1",
            document_uri="test://doc",
            document_title="Doc",
            page_numbers=[],
            headings=None,
            content="some content",
        )
        state = SessionState(citations_history=[[citation]])
        data = state.model_dump(mode="json")
        restored = SessionState.model_validate(data)
        assert len(restored.citations_history) == 1
        assert len(restored.citations_history[0]) == 1
        assert restored.citations_history[0][0].chunk_id == "c1"


class TestComputeStateDelta:
    """Tests for compute_state_delta."""

    def test_returns_delta_on_change(self):
        """compute_state_delta returns StateDeltaEvent when state changed."""
        old = SessionState()
        new = SessionState(citation_registry={"chunk-a": 1})

        result = compute_state_delta(old, new)

        assert isinstance(result, StateDeltaEvent)
        assert result.type == EventType.STATE_DELTA
        assert len(result.delta) > 0

    def test_returns_none_on_no_change(self):
        """compute_state_delta returns None when states are identical."""
        state = SessionState(document_filter=["doc1"])

        result = compute_state_delta(state, state.model_copy(deep=True))

        assert result is None

    def test_with_state_key(self):
        """compute_state_delta wraps delta under state_key."""
        old = SessionState()
        new = SessionState(document_filter=["doc1"])

        result = compute_state_delta(old, new, state_key="my.key")

        assert isinstance(result, StateDeltaEvent)
        # The delta paths should be prefixed with /my.key/
        paths = [op["path"] for op in result.delta]
        assert all(p.startswith("/my.key/") for p in paths)


class TestComputeCombinedStateDelta:
    """Tests for compute_combined_state_delta."""

    def test_returns_delta_on_change(self):
        """compute_combined_state_delta returns StateDeltaEvent when snapshots differ."""
        old = {"citations": []}
        new = {"citations": [{"index": 1, "chunk_id": "c1"}]}

        result = compute_combined_state_delta(old, new)

        assert isinstance(result, StateDeltaEvent)
        assert result.type == EventType.STATE_DELTA

    def test_returns_none_on_no_change(self):
        """compute_combined_state_delta returns None when snapshots are identical."""
        snapshot = {"citations": [], "document_filter": []}

        result = compute_combined_state_delta(snapshot, snapshot.copy())

        assert result is None

    def test_with_state_key_wraps(self):
        """compute_combined_state_delta wraps under state_key."""
        old = {"value": 1}
        new = {"value": 2}

        result = compute_combined_state_delta(old, new, state_key="ns")

        assert isinstance(result, StateDeltaEvent)
        paths = [op["path"] for op in result.delta]
        assert all(p.startswith("/ns/") for p in paths)

    def test_without_state_key(self):
        """compute_combined_state_delta works without state_key."""
        old = {"value": 1}
        new = {"value": 2}

        result = compute_combined_state_delta(old, new)

        assert isinstance(result, StateDeltaEvent)
        paths = [op["path"] for op in result.delta]
        assert any(p == "/value" for p in paths)
