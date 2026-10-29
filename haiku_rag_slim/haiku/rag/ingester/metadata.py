from collections.abc import Awaitable, Callable, Iterable, Mapping
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from haiku.rag.config.models import MetadataProviderConfig

if TYPE_CHECKING:
    from haiku.rag.sources.base import FetchResult

ENTRY_POINT_GROUP = "haiku.rag.metadata_providers"


@runtime_checkable
class MetadataProvider(Protocol):
    """Computes per-document metadata for the ingester. A package registers a
    factory under the ``haiku.rag.metadata_providers`` entry-point group; the
    factory takes the source's provider options as keyword arguments and
    returns an instance whose ``__call__`` the ingester invokes per job with
    the document's source id, uri, and fetched content."""

    async def __call__(
        self, source_id: str, uri: str, result: "FetchResult"
    ) -> dict: ...


MetadataProviderFactory = Callable[..., MetadataProvider]

# A provider with its source id bound: called with the uri and fetched content.
BoundMetadataProvider = Callable[[str, "FetchResult"], Awaitable[dict]]


@runtime_checkable
class LoadableEntryPoint(Protocol):
    """The slice of ``importlib.metadata.EntryPoint`` ``build_providers`` needs:
    a deferred ``load()`` returning the provider factory."""

    def load(self) -> MetadataProviderFactory: ...


def load_metadata_providers() -> dict[str, LoadableEntryPoint]:
    """Discover registered metadata-provider entry points, keyed by name. The
    entry points are not imported here; ``build_providers`` loads only the ones
    a source references, so an unused provider with a broken import does not
    fail the ingester at startup."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def build_providers(
    sources: Iterable[tuple[str, str | MetadataProviderConfig | None]],
    discovered: Mapping[str, LoadableEntryPoint],
) -> dict[str, MetadataProvider]:
    """Load and instantiate the provider configured for each source, passing it
    its options, keyed by source id. Sources with no provider are skipped, and
    only referenced entry points are loaded. Raises ValueError naming the
    source and provider for a name with no registered entry point, a module
    that fails to import, or a factory that fails."""
    providers: dict[str, MetadataProvider] = {}
    for source_id, entry in sources:
        if entry is None:
            continue
        if isinstance(entry, str):
            entry = MetadataProviderConfig(name=entry)
        name = entry.name
        try:
            entry_point = discovered[name]
        except KeyError:
            raise ValueError(
                f"Source {source_id!r} references unknown metadata provider "
                f"{name!r}; no entry point registered under {ENTRY_POINT_GROUP!r}."
            ) from None
        try:
            factory: MetadataProviderFactory = entry_point.load()
            providers[source_id] = factory(**entry.options)
        except Exception as exc:
            raise ValueError(
                f"Source {source_id!r} metadata provider {name!r}: {exc}"
            ) from exc
    return providers
