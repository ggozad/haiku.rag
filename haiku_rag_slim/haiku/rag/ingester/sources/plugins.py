from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from haiku.rag.ingester.sources.base import Source


@runtime_checkable
class SourceFactory(Protocol):
    """Builds a Source from a plugin source's config. A package registers a
    factory under the ``haiku.rag.sources`` entry-point group; the ingester
    calls it with the source id, the config's opaque ``options`` (which the
    plugin validates itself), and the ambient extension/size limits."""

    def __call__(
        self,
        *,
        source_id: str,
        options: dict[str, Any],
        supported_extensions: list[str] | None,
        max_file_size: int | None,
    ) -> Source: ...


@runtime_checkable
class LoadableEntryPoint(Protocol):
    """The slice of ``importlib.metadata.EntryPoint`` the factory loader needs:
    a deferred ``load()`` returning the source factory."""

    def load(self) -> SourceFactory: ...


ENTRY_POINT_GROUP = "haiku.rag.sources"


def load_source_factories() -> dict[str, LoadableEntryPoint]:
    """Discover registered source-factory entry points, keyed by name. The
    entry points are not imported here; ``build_source`` loads only the one a
    source references, so an unused plugin with a broken import does not fail
    the ingester at startup."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


__all__ = [
    "ENTRY_POINT_GROUP",
    "LoadableEntryPoint",
    "SourceFactory",
    "load_source_factories",
]
