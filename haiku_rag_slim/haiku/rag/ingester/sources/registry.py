from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from haiku.rag.ingester.sources.base import Source
from haiku.rag.ingester.sources.fs import FSSource
from haiku.rag.ingester.sources.http import HTTPSource
from haiku.rag.ingester.sources.s3 import S3Source


def resolve_fetcher(
    uri: str,
    sources: Iterable[Source] | None = None,
    *,
    storage_options: dict[str, str] | None = None,
) -> Source:
    """Pick a Source adapter for ``uri``.

    Configured ``sources`` win — the first whose ``supports(uri)`` returns True
    is returned. Without a configured match, an ad-hoc adapter is built from
    the URI scheme so one-shot calls (``add-src <uri>``) work without any
    configuration.
    """
    if sources:
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
            raise ValueError(f"Invalid S3 URI: {uri}")
        return S3Source(uri=f"s3://{bucket}/", storage_options=storage_options)

    raise ValueError(f"No source adapter for URI scheme {scheme!r}: {uri}")
