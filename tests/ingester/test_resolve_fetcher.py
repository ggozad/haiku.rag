from pathlib import Path

import pytest

from haiku.rag.ingester.sources import (
    resolve_adhoc_fetcher,
    resolve_configured_source,
)
from haiku.rag.ingester.sources.fs import FSSource
from haiku.rag.ingester.sources.http import HTTPSource
from haiku.rag.ingester.sources.s3 import S3Source

# --- resolve_adhoc_fetcher ---


def test_adhoc_resolves_fs_for_file_uri():
    src = resolve_adhoc_fetcher("file:///tmp/sample.md")
    assert isinstance(src, FSSource)


def test_adhoc_resolves_fs_for_bare_path():
    src = resolve_adhoc_fetcher("/tmp/sample.md")
    assert isinstance(src, FSSource)


def test_adhoc_resolves_http():
    src = resolve_adhoc_fetcher("https://example.com/x.pdf")
    assert isinstance(src, HTTPSource)


def test_adhoc_resolves_s3_scopes_to_bucket():
    src = resolve_adhoc_fetcher("s3://my-bucket/key.pdf")
    assert isinstance(src, S3Source)
    assert src.bucket == "my-bucket"
    assert src.prefix == ""


def test_adhoc_resolves_s3_forwards_storage_options():
    src = resolve_adhoc_fetcher(
        "s3://my-bucket/key.pdf",
        storage_options={"endpoint": "http://seaweed:8333", "allow_http": "true"},
    )
    assert isinstance(src, S3Source)
    assert src.storage_options["endpoint"] == "http://seaweed:8333"


def test_adhoc_unknown_scheme_raises():
    with pytest.raises(ValueError, match="No source adapter"):
        resolve_adhoc_fetcher("ftp://example.com/x")


def test_adhoc_configured_source_matches_first(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi")
    fs = FSSource(root=tmp_path)
    chosen = resolve_adhoc_fetcher((tmp_path / "a.md").as_uri(), sources=[fs])
    assert chosen is fs


def test_adhoc_configured_source_falls_through_when_no_match():
    fs = FSSource(root=Path("/tmp"))
    # https doesn't match an FS source — fall back to ad-hoc HTTPSource
    chosen = resolve_adhoc_fetcher("https://example.com/x", sources=[fs])
    assert isinstance(chosen, HTTPSource)
    assert chosen is not fs


def test_adhoc_returns_first_supporting_source_with_multiple_http():
    """Without source_id (ad-hoc mode), the first configured HTTP source
    that supports the URI wins. Used by `add-src` from the CLI."""
    arxiv = HTTPSource(source_id="arxiv", headers={"Authorization": "Bearer A"})
    intranet = HTTPSource(source_id="intranet", headers={"Authorization": "Bearer B"})
    chosen = resolve_adhoc_fetcher("https://example.com/x", sources=[arxiv, intranet])
    assert chosen is arxiv


# --- resolve_configured_source ---


def test_configured_picks_by_source_id():
    """Worker path: pick the source by id so credentials/headers of the
    configured source are reused instead of falling to whichever matched
    supports(uri) first."""
    arxiv = HTTPSource(source_id="arxiv", headers={"Authorization": "Bearer A"})
    intranet = HTTPSource(source_id="intranet", headers={"Authorization": "Bearer B"})
    assert (
        resolve_configured_source(
            "https://example.com/x", "intranet", [arxiv, intranet]
        )
        is intranet
    )


def test_configured_raises_when_matched_source_rejects_uri():
    """source_id selects by identity, but the source still has to support
    the URI. Mismatch = stale enqueue from a reconfigured source."""
    fs = FSSource(root=Path("/tmp"), source_id="local")
    with pytest.raises(ValueError, match="doesn't support URI"):
        resolve_configured_source("https://example.com/x", "local", [fs])


def test_configured_raises_when_no_source_matches():
    """Worker job carries a source_id from when it was enqueued. If the
    operator renamed/removed that source, falling back to an ad-hoc adapter
    would silently drop credentials. Raise instead so the job DLQs."""
    arxiv = HTTPSource(source_id="arxiv", headers={"Authorization": "Bearer A"})
    with pytest.raises(ValueError, match="No configured source with id 'gone'"):
        resolve_configured_source("https://example.com/x", "gone", [arxiv])


def test_configured_raises_when_sources_list_is_none():
    with pytest.raises(ValueError, match="No configured source with id"):
        resolve_configured_source("https://example.com/x", "arxiv", None)
