from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.ingester.sources.base import Source
from haiku.rag.ingester.sources.fs import FSSource
from haiku.rag.ingester.sources.http import HTTPSource
from haiku.rag.ingester.sources.s3 import S3Source


def resolve_fetcher(
    uri: str,
    sources: Iterable[Source] | None = None,
    *,
    source_id: str | None = None,
    storage_options: dict[str, str] | None = None,
) -> Source:
    """Pick a Source adapter for ``uri``.

    When ``source_id`` is given (worker path), the source with that id is
    used so credentials/headers of the configured source are reused rather
    than picking whichever source happens to match ``supports(uri)`` first.
    Without ``source_id`` (ad-hoc ``add-src <uri>``), the first configured
    source whose ``supports(uri)`` returns True wins, then a scheme-based
    ad-hoc adapter.
    """
    if sources:
        if source_id is not None:
            for src in sources:
                if src.source_id == source_id:
                    if not src.supports(uri):
                        raise UnsupportedSourceError(
                            f"Source {source_id!r} doesn't support URI {uri!r}"
                        )
                    return src
        else:
            for src in sources:
                if src.supports(uri):
                    return src

    scheme = urlparse(uri).scheme
    if scheme in ("", "file"):
        # Root only matters for discover(); fetch() needs an absolute path
        # that already encodes the location, so any root is correct.
        return FSSource(root=Path("/"))
    if scheme in ("http", "https"):
        return HTTPSource(source_id="http:adhoc")
    if scheme == "s3":
        bucket = urlparse(uri).netloc
        if not bucket:
            raise UnsupportedSourceError(f"Invalid S3 URI: {uri}")
        return S3Source(uri=f"s3://{bucket}/", storage_options=storage_options)

    raise UnsupportedSourceError(f"No source adapter for URI scheme {scheme!r}: {uri}")
