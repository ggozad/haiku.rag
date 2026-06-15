from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

ENTRY_POINT_GROUP = "haiku.rag.metadata_providers"


@runtime_checkable
class MetadataProvider(Protocol):
    """Computes per-document metadata for the ingester. A package registers a
    zero-arg factory under the ``haiku.rag.metadata_providers`` entry-point
    group; the factory returns an instance whose ``__call__`` the ingester
    invokes per job with the document's source id and uri."""

    async def __call__(self, source_id: str, uri: str) -> dict: ...


MetadataProviderFactory = Callable[[], MetadataProvider]


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
    sources: Iterable[tuple[str, str | None]],
    discovered: Mapping[str, LoadableEntryPoint],
) -> dict[str, MetadataProvider]:
    """Load and instantiate the provider named by each ``(source_id, name)``
    pair, keyed by source id. Pairs with no name are skipped, and only
    referenced entry points are loaded. Raises ValueError if a name has no
    registered entry point so a misconfigured source fails at startup rather
    than silently dropping metadata."""
    providers: dict[str, MetadataProvider] = {}
    for source_id, name in sources:
        if name is None:
            continue
        try:
            entry_point = discovered[name]
        except KeyError:
            raise ValueError(
                f"Source {source_id!r} references unknown metadata provider "
                f"{name!r}; no entry point registered under {ENTRY_POINT_GROUP!r}."
            ) from None
        factory: MetadataProviderFactory = entry_point.load()
        providers[source_id] = factory()
    return providers
