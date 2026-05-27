from pathlib import Path

import pytest

from haiku.rag.ingester.sources import resolve_fetcher
from haiku.rag.ingester.sources.fs import FSSource
from haiku.rag.ingester.sources.http import HTTPSource
from haiku.rag.ingester.sources.s3 import S3Source


def test_resolves_fs_for_file_uri():
    src = resolve_fetcher("file:///tmp/sample.md")
    assert isinstance(src, FSSource)


def test_resolves_fs_for_bare_path():
    src = resolve_fetcher("/tmp/sample.md")
    assert isinstance(src, FSSource)


def test_resolves_http():
    src = resolve_fetcher("https://example.com/x.pdf")
    assert isinstance(src, HTTPSource)


def test_resolves_s3_scopes_to_bucket():
    src = resolve_fetcher("s3://my-bucket/key.pdf")
    assert isinstance(src, S3Source)
    assert src.bucket == "my-bucket"
    assert src.prefix == ""


def test_resolves_s3_forwards_storage_options():
    src = resolve_fetcher(
        "s3://my-bucket/key.pdf",
        storage_options={"endpoint": "http://seaweed:8333", "allow_http": "true"},
    )
    assert isinstance(src, S3Source)
    assert src.storage_options["endpoint"] == "http://seaweed:8333"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="No source adapter"):
        resolve_fetcher("ftp://example.com/x")


def test_configured_source_matches_first(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi")
    fs = FSSource(root=tmp_path)
    chosen = resolve_fetcher((tmp_path / "a.md").as_uri(), sources=[fs])
    assert chosen is fs


def test_configured_source_falls_through_when_no_match():
    fs = FSSource(root=Path("/tmp"))
    # https doesn't match an FS source — fall back to ad-hoc HTTPSource
    chosen = resolve_fetcher("https://example.com/x", sources=[fs])
    assert isinstance(chosen, HTTPSource)
    assert chosen is not fs


def test_source_id_prefers_matching_source_over_first_supports():
    """Two HTTPSource configs with different auth headers: passing
    source_id picks the right one. Without source_id, the first one to
    return True from supports(uri) wins regardless of which config
    actually owns the job."""
    arxiv = HTTPSource(source_id="arxiv", headers={"Authorization": "Bearer A"})
    intranet = HTTPSource(source_id="intranet", headers={"Authorization": "Bearer B"})

    by_id = resolve_fetcher(
        "https://example.com/x", sources=[arxiv, intranet], source_id="intranet"
    )
    assert by_id is intranet

    without_id = resolve_fetcher("https://example.com/x", sources=[arxiv, intranet])
    assert without_id is arxiv


def test_source_id_raises_when_matched_source_rejects_uri():
    """source_id selects the source by identity, but that source still
    has to support the URI. Mismatch means a stale enqueue from a
    reconfigured source — surface it instead of silently picking another."""
    fs = FSSource(root=Path("/tmp"), source_id="local")
    with pytest.raises(ValueError, match="doesn't support URI"):
        resolve_fetcher("https://example.com/x", sources=[fs], source_id="local")
