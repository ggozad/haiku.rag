from pydantic import BaseModel

from haiku.rag.tools.context import ToolContext, prepare_context
from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState


class TestState(BaseModel):
    value: int = 0


class TestStateWithList(BaseModel):
    items: list[str] = []


def test_tool_context_defaults():
    """Test ToolContext has sensible defaults."""
    ctx = ToolContext()
    assert ctx._namespaces == {}


def test_register_and_get():
    """Test register and get state for a namespace."""
    ctx = ToolContext()
    state = TestState(value=42)

    ctx.register("test.namespace", state)

    retrieved = ctx.get("test.namespace")
    assert retrieved is state
    assert retrieved.value == 42


def test_get_nonexistent_namespace():
    """Test get returns None for unregistered namespace."""
    ctx = ToolContext()
    assert ctx.get("nonexistent") is None


def test_get_or_create_creates_new():
    """Test get_or_create creates state when namespace doesn't exist."""
    ctx = ToolContext()

    state = ctx.get_or_create("test.namespace", TestStateWithList)
    assert isinstance(state, TestStateWithList)
    assert state.items == []


def test_get_or_create_returns_existing():
    """Test get_or_create returns existing state."""
    ctx = ToolContext()

    state1 = ctx.get_or_create("test.namespace", TestStateWithList)
    state1.items.append("item1")

    state2 = ctx.get_or_create("test.namespace", TestStateWithList)
    assert state2 is state1
    assert state2.items == ["item1"]


def test_clear_namespace():
    """Test clear_namespace removes only the specified namespace."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=1))
    ctx.register("ns2", TestState(value=2))

    ctx.clear_namespace("ns1")

    assert ctx.get("ns1") is None
    ns2 = ctx.get("ns2")
    assert isinstance(ns2, TestState)
    assert ns2.value == 2


def test_clear_namespace_nonexistent():
    """Test clear_namespace handles nonexistent namespace gracefully."""
    ctx = ToolContext()
    ctx.clear_namespace("nonexistent")  # Should not raise


def test_clear_all():
    """Test clear_all clears all namespaces."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=1))
    ctx.register("ns2", TestState(value=2))

    ctx.clear_all()

    assert ctx.get("ns1") is None
    assert ctx.get("ns2") is None


def test_namespaces_property():
    """Test namespaces property lists all registered namespaces."""
    ctx = ToolContext()
    assert ctx.namespaces == []

    ctx.register("ns1", TestState())
    ctx.register("ns2", TestState())

    assert set(ctx.namespaces) == {"ns1", "ns2"}


def test_shared_namespace_between_toolsets():
    """Test that toolsets can share state via the same namespace."""

    class SharedState(BaseModel):
        citations: dict[str, int] = {}

    SHARED_NAMESPACE = "haiku.rag.citations"

    ctx = ToolContext()

    # First toolset registers the shared state
    state1 = ctx.get_or_create(SHARED_NAMESPACE, SharedState)
    state1.citations["chunk-a"] = 1

    # Second toolset gets the same state
    state2 = ctx.get_or_create(SHARED_NAMESPACE, SharedState)
    assert state2 is state1
    assert state2.citations == {"chunk-a": 1}

    # Both see updates
    state2.citations["chunk-b"] = 2
    assert state1.citations == {"chunk-a": 1, "chunk-b": 2}


def test_dump_namespaces():
    """Test dump_namespaces serializes all registered states."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=10))
    ctx.register("ns2", TestStateWithList(items=["a", "b"]))

    data = ctx.dump_namespaces()

    assert data == {
        "ns1": {"value": 10},
        "ns2": {"items": ["a", "b"]},
    }


def test_load_namespace():
    """Test load_namespace deserializes and registers state."""
    ctx = ToolContext()

    state = ctx.load_namespace("ns1", TestState, {"value": 42})

    assert isinstance(state, TestState)
    assert state.value == 42
    assert ctx.get("ns1") is state


def test_serialization_roundtrip():
    """Test full serialization/deserialization roundtrip."""
    # Create and populate context
    original = ToolContext()
    original.register("search", TestStateWithList(items=["result1", "result2"]))
    original.register("qa", TestState(value=99))

    # Serialize
    ns_data = original.dump_namespaces()

    # Deserialize
    restored = ToolContext()
    restored.load_namespace("search", TestStateWithList, ns_data["search"])
    restored.load_namespace("qa", TestState, ns_data["qa"])

    # Verify
    search_state = restored.get("search")
    assert isinstance(search_state, TestStateWithList)
    assert search_state.items == ["result1", "result2"]

    qa_state = restored.get("qa")
    assert isinstance(qa_state, TestState)
    assert qa_state.value == 99


def test_get_with_type_match():
    """Test get with state_type returns typed state when type matches."""
    ctx = ToolContext()
    state = TestState(value=42)
    ctx.register("ns", state)

    result = ctx.get("ns", TestState)
    assert result is state
    assert result.value == 42


def test_get_with_type_mismatch():
    """Test get with state_type returns None when type doesn't match."""
    ctx = ToolContext()
    ctx.register("ns", TestState(value=42))

    result = ctx.get("ns", TestStateWithList)
    assert result is None


def test_get_without_type():
    """Test get without state_type returns BaseModel (unchanged behavior)."""
    ctx = ToolContext()
    state = TestState(value=42)
    ctx.register("ns", state)

    result = ctx.get("ns")
    assert result is state


def test_tool_context_state_key_default_none():
    """Test ToolContext state_key defaults to None."""
    ctx = ToolContext()
    assert ctx.state_key is None


def test_tool_context_state_key_set():
    """Test ToolContext state_key can be set."""
    ctx = ToolContext()
    ctx.state_key = "haiku.rag.chat"
    assert ctx.state_key == "haiku.rag.chat"


# =============================================================================
# ToolContextCache Tests
# =============================================================================


def test_tool_context_cache_get_or_create_new():
    """Test get_or_create returns a new context with is_new=True."""
    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache()
    context, is_new = cache.get_or_create("thread-1")

    assert isinstance(context, ToolContext)
    assert is_new is True


def test_tool_context_cache_get_or_create_existing():
    """Test get_or_create returns existing context with is_new=False."""
    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache()
    ctx1, is_new1 = cache.get_or_create("thread-1")
    ctx2, is_new2 = cache.get_or_create("thread-1")

    assert ctx2 is ctx1
    assert is_new1 is True
    assert is_new2 is False


def test_tool_context_cache_different_keys():
    """Test get_or_create returns different contexts for different keys."""
    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache()
    ctx1, _ = cache.get_or_create("thread-1")
    ctx2, _ = cache.get_or_create("thread-2")

    assert ctx1 is not ctx2


def test_tool_context_cache_ttl_expiry():
    """Test that contexts are evicted after TTL expires."""
    from datetime import timedelta

    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache(ttl=timedelta(seconds=0))
    ctx1, _ = cache.get_or_create("thread-1")

    # With zero TTL, next access should create a new context
    ctx2, is_new = cache.get_or_create("thread-1")

    assert ctx2 is not ctx1
    assert is_new is True


def test_tool_context_cache_remove():
    """Test remove deletes a specific key."""
    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache()
    cache.get_or_create("thread-1")
    cache.get_or_create("thread-2")

    cache.remove("thread-1")

    ctx, is_new = cache.get_or_create("thread-1")
    assert is_new is True

    # thread-2 should still exist
    ctx2, is_new2 = cache.get_or_create("thread-2")
    assert is_new2 is False


def test_tool_context_cache_remove_nonexistent():
    """Test remove handles nonexistent key gracefully."""
    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache()
    cache.remove("nonexistent")  # Should not raise


def test_tool_context_cache_clear():
    """Test clear removes all entries."""
    from haiku.rag.tools.context import ToolContextCache

    cache = ToolContextCache()
    cache.get_or_create("thread-1")
    cache.get_or_create("thread-2")

    cache.clear()

    ctx1, is_new1 = cache.get_or_create("thread-1")
    ctx2, is_new2 = cache.get_or_create("thread-2")
    assert is_new1 is True
    assert is_new2 is True


# =============================================================================
# build_state_snapshot / restore_state_snapshot Tests
# =============================================================================


class NestedModel(BaseModel):
    name: str = ""
    count: int = 0


class StateWithNested(BaseModel):
    nested: NestedModel | None = None
    tags: list[str] = []


def test_build_state_snapshot_empty():
    """build_state_snapshot on empty context returns empty dict."""
    ctx = ToolContext()
    assert ctx.build_state_snapshot() == {}


def test_build_state_snapshot_single_namespace():
    """build_state_snapshot with one namespace returns its fields."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=42))
    snapshot = ctx.build_state_snapshot()
    assert snapshot == {"value": 42}


def test_build_state_snapshot_multiple_namespaces():
    """build_state_snapshot merges fields from all namespaces."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=42))
    ctx.register("ns2", TestStateWithList(items=["a", "b"]))
    snapshot = ctx.build_state_snapshot()
    assert snapshot == {"value": 42, "items": ["a", "b"]}


def test_build_state_snapshot_nested_model():
    """build_state_snapshot serializes nested models with mode='json'."""
    from datetime import datetime

    class TimestampState(BaseModel):
        ts: datetime | None = None

    ctx = ToolContext()
    ctx.register("ns", TimestampState(ts=datetime(2025, 1, 27, 12, 0, 0)))
    snapshot = ctx.build_state_snapshot()
    assert isinstance(snapshot["ts"], str)
    assert snapshot["ts"] == "2025-01-27T12:00:00"


def test_restore_state_snapshot_empty_context():
    """restore_state_snapshot on empty context is a no-op."""
    ctx = ToolContext()
    ctx.restore_state_snapshot({"value": 42})
    assert ctx.namespaces == []


def test_restore_state_snapshot_single_namespace():
    """restore_state_snapshot updates matching fields in registered namespaces."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=0))
    ctx.restore_state_snapshot({"value": 99})
    state = ctx.get("ns1", TestState)
    assert state is not None
    assert state.value == 99


def test_restore_state_snapshot_partial_update():
    """restore_state_snapshot only touches fields present in data."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=42))
    ctx.register("ns2", TestStateWithList(items=["original"]))
    # Only update ns2's items, not ns1's value
    ctx.restore_state_snapshot({"items": ["updated"]})
    ns1 = ctx.get("ns1", TestState)
    ns2 = ctx.get("ns2", TestStateWithList)
    assert ns1 is not None
    assert ns2 is not None
    assert ns1.value == 42
    assert ns2.items == ["updated"]


def test_restore_state_snapshot_nested_model():
    """restore_state_snapshot deserializes nested models from dicts."""
    ctx = ToolContext()
    ctx.register("ns", StateWithNested())
    ctx.restore_state_snapshot({"nested": {"name": "foo", "count": 5}, "tags": ["x"]})
    state = ctx.get("ns", StateWithNested)
    assert state is not None
    assert state.nested is not None
    assert state.nested.name == "foo"
    assert state.nested.count == 5
    assert state.tags == ["x"]


def test_state_snapshot_roundtrip():
    """build then restore produces equivalent state."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=42))
    ctx.register("ns2", TestStateWithList(items=["a", "b"]))

    snapshot = ctx.build_state_snapshot()

    ctx2 = ToolContext()
    ctx2.register("ns1", TestState())
    ctx2.register("ns2", TestStateWithList())
    ctx2.restore_state_snapshot(snapshot)

    ns1 = ctx2.get("ns1", TestState)
    ns2 = ctx2.get("ns2", TestStateWithList)
    assert ns1 is not None
    assert ns2 is not None
    assert ns1.value == 42
    assert ns2.items == ["a", "b"]


def test_restore_state_snapshot_ignores_unknown_fields():
    """restore_state_snapshot ignores fields not in any registered namespace."""
    ctx = ToolContext()
    ctx.register("ns1", TestState(value=0))
    ctx.restore_state_snapshot({"value": 10, "unknown_field": "ignored"})
    ns1 = ctx.get("ns1", TestState)
    assert ns1 is not None
    assert ns1.value == 10


# --- prepare_context tests ---


def test_prepare_context_default_features():
    """Default features register SessionState only."""
    ctx = ToolContext()
    prepare_context(ctx)
    assert ctx.get(SESSION_NAMESPACE, SessionState) is not None
    assert ctx.get(QA_SESSION_NAMESPACE, QASessionState) is None


def test_prepare_context_with_qa():
    """QA feature registers both SessionState and QASessionState."""
    ctx = ToolContext()
    prepare_context(ctx, features=["search", "qa"])
    assert ctx.get(SESSION_NAMESPACE, SessionState) is not None
    assert ctx.get(QA_SESSION_NAMESPACE, QASessionState) is not None


def test_prepare_context_sets_state_key():
    """state_key is set on context when provided."""
    ctx = ToolContext()
    prepare_context(ctx, state_key="my_app")
    assert ctx.state_key == "my_app"


def test_prepare_context_no_state_key_by_default():
    """state_key is not set when not provided."""
    ctx = ToolContext()
    prepare_context(ctx)
    assert ctx.state_key is None


def test_prepare_context_idempotent():
    """Calling prepare_context twice doesn't create duplicate state."""
    ctx = ToolContext()
    prepare_context(ctx, features=["search", "qa"])
    session1 = ctx.get(SESSION_NAMESPACE, SessionState)
    qa1 = ctx.get(QA_SESSION_NAMESPACE, QASessionState)
    prepare_context(ctx, features=["search", "qa"])
    assert ctx.get(SESSION_NAMESPACE, SessionState) is session1
    assert ctx.get(QA_SESSION_NAMESPACE, QASessionState) is qa1
