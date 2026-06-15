from collections.abc import Callable
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


def load_metadata_providers() -> dict[str, MetadataProviderFactory]:
    """Discover registered metadata-provider factories, keyed by entry-point name."""
    return {ep.name: ep.load() for ep in entry_points(group=ENTRY_POINT_GROUP)}
