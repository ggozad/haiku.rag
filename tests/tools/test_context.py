from pydantic import BaseModel

from haiku.rag.tools.context import ToolContext


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
