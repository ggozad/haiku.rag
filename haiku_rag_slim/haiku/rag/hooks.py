import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Literal, Protocol, overload, runtime_checkable

from haiku.rag.store.models.chunk import SearchResult, SearchType
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.client import HaikuRAG

ENTRY_POINT_GROUP = "haiku.rag.hooks"

logger = logging.getLogger(__name__)

IngestOperation = Literal["create", "update"]


@dataclass
class IngestEvent:
    """Documents whose content was written in one operation. Batch imports
    carry the whole batch in a single event."""

    documents: list[Document]
    operation: IngestOperation


@dataclass
class DeleteEvent:
    """Documents removed in one operation. A cascade delete carries the root
    and all its children in a single event. The documents no longer exist in
    the database; the models are the last-known state."""

    documents: list[Document]


@dataclass
class SearchRequest:
    """The parameters a search will run with. ``before_search`` hooks may
    modify ``query``, ``filter``, ``search_type``, and ``limit``."""

    query: "str | bytes | PILImage.Image"
    filter: str | None
    search_type: SearchType | None
    limit: int


class Hook:
    """Base class for client lifecycle hooks. Subclasses override any subset.

    A package registers a zero-arg factory under the ``haiku.rag.hooks``
    entry-point group; ``config.hooks`` lists the hooks to activate, and they
    run in the listed order at every hook point. Hooks receive the ``HaikuRAG``
    client, so they may search, read repositories, or keep their own state in
    the database via ``client.store`` (table names must use the ``hook_``
    prefix to stay clear of core tables and migrations).
    """

    async def after_ingest(self, client: "HaikuRAG", event: IngestEvent) -> None:
        """Content was written for ``event.documents``. ``event.operation``
        is ``"create"`` for new documents and ``"update"`` when an existing
        document's content was rewritten (including creation against an
        already-stored URI). Replace any state derived from the documents
        regardless of the operation: even a creation may be a retry.
        Metadata/title-only updates do not fire.

        Best-effort observer: the operation has already committed, so
        exceptions are logged and never raised, and subsequent hooks still
        run. Correctness-critical derived state needs its own
        reconciliation. Post-commit hooks are not a supported transformation
        point: mutating event models does not alter the committed record,
        and explicit client writes are separate operations, not atomic with
        the original write."""

    async def after_delete(self, client: "HaikuRAG", event: DeleteEvent) -> None:
        """``event.documents`` were deleted; cascades arrive as one event.
        Best-effort observer with the same contract as ``after_ingest``."""

    async def before_search(
        self, client: "HaikuRAG", request: SearchRequest
    ) -> SearchRequest:
        """Transform the search parameters before retrieval. Text queries
        only; the returned request's query feeds both the vector and FTS
        sides."""
        return request

    async def after_search(
        self,
        client: "HaikuRAG",
        request: SearchRequest,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Transform or annotate search results before they are returned.
        ``request`` reflects any ``before_search`` transformations."""
        return results


@overload
async def notify(
    hooks: Sequence[Hook],
    method: Literal["after_ingest"],
    client: "HaikuRAG",
    event: IngestEvent,
) -> None: ...


@overload
async def notify(
    hooks: Sequence[Hook],
    method: Literal["after_delete"],
    client: "HaikuRAG",
    event: DeleteEvent,
) -> None: ...


async def notify(
    hooks: Sequence[Hook],
    method: Literal["after_ingest", "after_delete"],
    client: "HaikuRAG",
    event: IngestEvent | DeleteEvent,
) -> None:
    """Fire post-commit observer hooks best-effort: a hook failure is logged
    and never raised (the operation already committed), and subsequent hooks
    still run."""
    for hook in hooks:
        try:
            await getattr(hook, method)(client, event)
        except Exception:
            cls = type(hook)
            logger.exception(
                "%s.%s.%s failed for documents %s",
                cls.__module__,
                cls.__qualname__,
                method,
                [d.id for d in event.documents],
            )


HookFactory = Callable[[], Hook]


@runtime_checkable
class LoadableEntryPoint(Protocol):
    """The slice of ``importlib.metadata.EntryPoint`` ``build_hooks`` needs:
    a deferred ``load()`` returning the hook factory."""

    def load(self) -> HookFactory: ...


def load_hooks() -> dict[str, LoadableEntryPoint]:
    """Discover registered hook entry points, keyed by name. The entry points
    are not imported here; ``build_hooks`` loads only the ones the config
    references, so an unused hook with a broken import does not fail client
    construction."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def build_hooks(
    names: Sequence[str],
    discovered: Mapping[str, LoadableEntryPoint],
) -> list[Hook]:
    """Load and instantiate the named hooks in configured order. Raises
    ValueError for a name with no registered entry point so a misconfigured
    client fails at construction rather than silently skipping a hook."""
    hooks: list[Hook] = []
    for name in names:
        try:
            entry_point = discovered[name]
        except KeyError:
            raise ValueError(
                f"Config references unknown hook {name!r}; no entry point "
                f"registered under {ENTRY_POINT_GROUP!r}."
            ) from None
        factory: HookFactory = entry_point.load()
        hooks.append(factory())
    return hooks
