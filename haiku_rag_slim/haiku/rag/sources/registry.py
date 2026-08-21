from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.sources.base import Source
from haiku.rag.sources.fs import FSSource
from haiku.rag.sources.http import HTTPSource
from haiku.rag.sources.s3 import S3Source
from haiku.rag.uri import is_local_uri, uri_to_path


def resolve_configured_source(
    uri: str,
    source_id: str,
    sources: Iterable[Source] | None,
) -> Source:
    """Strict lookup: return the configured source with this id, or raise.

    Worker jobs carry source_id from when they were enqueued. Falling back
    to an ad-hoc fetcher would silently drop credentials when a source has
    been renamed or removed from config — better to raise and let the job
    DLQ so the misconfiguration surfaces.
    """
    for src in sources or ():
        if src.source_id == source_id:
            if not src.supports(uri):
                raise UnsupportedSourceError(
                    f"Source {source_id!r} doesn't support URI {uri!r}"
                )
            return src
    raise UnsupportedSourceError(
        f"No configured source with id {source_id!r} for URI {uri!r}"
    )


def resolve_adhoc_fetcher(
    uri: str,
    *,
    sources: Iterable[Source] | None = None,
    storage_options: dict[str, str] | None = None,
) -> Source:
    """Best-effort lookup for one-shot fetches (e.g. ``add-src <uri>``).

    Configured ``sources`` win when one matches; otherwise a scheme-based
    adapter is built so any URI can be fetched without configuration.
    """
    if sources:
        for src in sources:
            if src.supports(uri):
                return src

    if is_local_uri(uri):
        # Root only matters for discover(); fetch() needs an absolute path
        # that already encodes the location, so the path's own anchor is
        # enough. On Windows "/" is only the current drive.
        return FSSource(root=Path(uri_to_path(uri).anchor or "/"))

    scheme = urlparse(uri).scheme
    if scheme in ("http", "https"):
        return HTTPSource(source_id="http:adhoc")
    if scheme == "s3":
        bucket = urlparse(uri).netloc
        if not bucket:
            raise UnsupportedSourceError(f"Invalid S3 URI: {uri}")
        return S3Source(uri=f"s3://{bucket}/", storage_options=storage_options)

    raise UnsupportedSourceError(f"No source adapter for URI scheme {scheme!r}: {uri}")
