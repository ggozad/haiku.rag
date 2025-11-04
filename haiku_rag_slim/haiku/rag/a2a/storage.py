import logging
from collections import OrderedDict

try:
    from fasta2a.schema import Artifact, Message, TaskState  # type: ignore
    from fasta2a.storage import InMemoryStorage, Storage  # type: ignore
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e

logger = logging.getLogger(__name__)


class LRUMemoryStorage(Storage[list[Message]]):  # type: ignore
    """Storage wrapper with LRU eviction for contexts.

    Enforces a maximum context limit using LRU (Least Recently Used) eviction.
    """

    def __init__(self, storage: InMemoryStorage, max_contexts: int):
        self.storage = storage
        self.max_contexts = max_contexts
        # Track context access order (LRU cache)
        self.context_order: OrderedDict[str, None] = OrderedDict()

    async def load_context(self, context_id: str) -> list[Message] | None:
        """Load context and update access order."""
        result = await self.storage.load_context(context_id)
        if result is not None:
            # Move to end (most recently used)
            self.context_order.pop(context_id, None)
            self.context_order[context_id] = None
        return result

    async def update_context(self, context_id: str, context: list[Message]) -> None:
        """Update context and enforce LRU limit."""
        await self.storage.update_context(context_id, context)
        # Move to end (most recently used)
        self.context_order.pop(context_id, None)
        self.context_order[context_id] = None

        # Enforce max contexts limit (LRU eviction)
        while len(self.context_order) > self.max_contexts:
            # Remove oldest (first item in OrderedDict)
            oldest_context_id = next(iter(self.context_order))
            self.context_order.pop(oldest_context_id)
            logger.debug(
                f"Evicted context {oldest_context_id} (LRU, limit={self.max_contexts})"
            )

    async def load_task(self, task_id: str, history_length: int | None = None):
        """Delegate to underlying storage."""
        return await self.storage.load_task(task_id, history_length)

    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        new_artifacts: list[Artifact] | None = None,
        new_messages: list[Message] | None = None,
    ):
        """Delegate to underlying storage."""
        return await self.storage.update_task(
            task_id, state, new_artifacts, new_messages
        )

    async def submit_task(self, context_id: str, message: Message):
        """Delegate to underlying storage."""
        return await self.storage.submit_task(context_id, message)
