from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class RAGDeps(Protocol):
    """Contract for toolset dependencies injected via RunContext.

    Any deps object passed to an agent using haiku.rag toolsets must
    provide these attributes.
    """

    client: "HaikuRAG"
    tool_context: "ToolContext | None"


class ToolContext(BaseModel):
    """Generic state container for haiku.rag toolsets.

    Toolsets register their own Pydantic model state under namespaces.
    Multiple toolsets can share state by registering under the same namespace.

    All registered states must be Pydantic BaseModel subclasses, making
    the entire context serializable via model_dump()/model_validate().

    Example:
        # Define toolset-specific state
        class SearchState(BaseModel):
            results: list[SearchResult] = []
            filter: str | None = None

        SEARCH_NAMESPACE = "haiku.rag.search"

        # In toolset factory
        def create_search_toolset(config):
            async def search(ctx: RunContext[RAGDeps], query: str):
                tool_context = ctx.deps.tool_context
                if tool_context:
                    state = tool_context.get_or_create(SEARCH_NAMESPACE, SearchState)
                ...

        # Usage
        search_tools = create_search_toolset(config)
        agent = Agent(..., toolsets=[search_tools])
        await agent.run("...", deps=my_deps)

        # Access accumulated state
        search_state = context.get(SEARCH_NAMESPACE)
        for result in search_state.results:
            print(f"{result.document_title}")

        # Serialize entire context
        ns_data = context.dump_namespaces()
    """

    state_key: str | None = None
    _namespaces: dict[str, BaseModel] = PrivateAttr(default_factory=dict)

    def register(self, namespace: str, state: BaseModel) -> None:
        """Register state for a namespace.

        Args:
            namespace: Unique identifier for the toolset (e.g., "haiku.rag.search")
            state: A Pydantic BaseModel instance to store

        Overwrites any existing state for the namespace.
        """
        self._namespaces[namespace] = state

    @overload
    def get(self, namespace: str) -> BaseModel | None: ...

    @overload
    def get(self, namespace: str, state_type: type[T]) -> T | None: ...

    def get(
        self, namespace: str, state_type: type[T] | None = None
    ) -> BaseModel | T | None:
        """Get state for a namespace, or None if not registered.

        When state_type is provided, returns the state only if it matches
        the expected type, otherwise returns None.
        """
        state = self._namespaces.get(namespace)
        if state_type is not None:
            return state if isinstance(state, state_type) else None
        return state

    def get_or_create(self, namespace: str, state_type: type[T]) -> T:
        """Get state for a namespace, creating it if not registered.

        Args:
            namespace: The namespace to get or create state for.
            state_type: A Pydantic BaseModel subclass to instantiate if needed.

        Returns:
            The state for the namespace.
        """
        if namespace not in self._namespaces:
            self._namespaces[namespace] = state_type()
        return self._namespaces[namespace]  # type: ignore[return-value]

    def clear_namespace(self, namespace: str) -> None:
        """Clear state for a specific namespace."""
        if namespace in self._namespaces:
            del self._namespaces[namespace]

    def clear_all(self) -> None:
        """Clear all namespaces."""
        self._namespaces.clear()

    @property
    def namespaces(self) -> list[str]:
        """List all registered namespaces."""
        return list(self._namespaces.keys())

    def dump_namespaces(self) -> dict[str, dict[str, Any]]:
        """Serialize all namespace states to a dictionary.

        Returns:
            Dict mapping namespace -> serialized state dict.
        """
        return {ns: state.model_dump() for ns, state in self._namespaces.items()}

    def load_namespace(self, namespace: str, state_type: type[T], data: dict) -> T:
        """Deserialize and register state for a namespace.

        Args:
            namespace: The namespace to register the state under.
            state_type: The Pydantic model class to deserialize into.
            data: The serialized state data.

        Returns:
            The deserialized and registered state.
        """
        state = state_type.model_validate(data)
        self._namespaces[namespace] = state
        return state


class ToolContextCache:
    """In-memory cache for ToolContext instances, keyed by external session/thread ID."""

    def __init__(self, ttl: timedelta = timedelta(hours=1)) -> None:
        self._cache: dict[str, ToolContext] = {}
        self._timestamps: dict[str, datetime] = {}
        self._ttl = ttl

    def get_or_create(self, key: str) -> tuple[ToolContext, bool]:
        """Get an existing context or create a new one.

        Returns:
            Tuple of (context, is_new) where is_new is True if a new context was created.
        """
        self._cleanup()
        if key in self._cache:
            self._timestamps[key] = datetime.now()
            return self._cache[key], False

        context = ToolContext()
        self._cache[key] = context
        self._timestamps[key] = datetime.now()
        return context, True

    def remove(self, key: str) -> None:
        """Remove a specific key from the cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
        self._timestamps.clear()

    def _cleanup(self) -> None:
        """Remove entries older than TTL."""
        now = datetime.now()
        expired = [
            key for key, ts in self._timestamps.items() if (now - ts) >= self._ttl
        ]
        for key in expired:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
