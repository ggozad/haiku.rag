from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models.chunk import SearchResult
    from haiku.rag.store.models.document import Document

ENTRY_POINT_GROUP = "haiku.rag.hooks"


class Hook:
    """Base class for client lifecycle hooks. Subclasses override any subset.

    A package registers a zero-arg factory under the ``haiku.rag.hooks``
    entry-point group; ``config.hooks`` lists the hooks to activate, and they
    run in the listed order at every hook point. Hooks receive the ``HaikuRAG``
    client, so they may search, read repositories, or keep their own state in
    the database via ``client.store`` (table names must use the ``hook_``
    prefix to stay clear of core tables and migrations).
    """

    async def after_ingest(self, client: "HaikuRAG", document: "Document") -> None:
        """A document's content was written (create, import, batch import,
        update). Replace any state derived from this document: create and
        update are deliberately the same event. Metadata/title-only updates
        do not fire."""

    async def after_delete(self, client: "HaikuRAG", document_id: str) -> None:
        """A document was deleted; fires once per document in a cascade."""

    async def before_search(
        self, client: "HaikuRAG", query: str, filter: str | None
    ) -> tuple[str, str | None]:
        """Transform the query and/or filter before retrieval. Text queries
        only; the returned query feeds both the vector and FTS sides."""
        return query, filter

    async def after_search(
        self,
        client: "HaikuRAG",
        query: "str | bytes | PILImage.Image",
        results: "list[SearchResult]",
    ) -> "list[SearchResult]":
        """Transform or annotate search results before they are returned."""
        return results


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
