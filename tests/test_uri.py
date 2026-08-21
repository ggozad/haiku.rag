"""Tests for haiku.rag.uri."""

import os
from pathlib import Path, PureWindowsPath

import pytest

from haiku.rag.uri import is_local_uri, uri_to_path


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("file:///tmp/a.md", True),
        ("/tmp/a.md", True),
        ("relative/a.md", True),
        # urlparse reads the Windows drive letter as the scheme.
        ("C:/docs/a.pdf", True),
        ("c:\\docs\\a.pdf", True),
        ("x:content", False),
        ("http://example.com/a.pdf", False),
        ("https://example.com/a.pdf", False),
        ("s3://bucket/a.pdf", False),
        ("webdav://host/a.pdf", False),
    ],
)
def test_is_local_uri(uri, expected):
    assert is_local_uri(uri) is expected


def test_uri_to_path_decodes_percent_escapes():
    assert uri_to_path("file:///tmp/a%5Bb%5D%20c.md") == Path("/tmp/a[b] c.md")


def test_uri_to_path_leaves_bare_paths_alone():
    """A bare path is not a URI, so ``a%20b.md`` is a filename, not ``a b.md``."""
    assert uri_to_path("/tmp/a%20b.md") == Path("/tmp/a%20b.md")


def test_uri_to_path_rejects_remote_schemes():
    with pytest.raises(ValueError, match="Not a local URI"):
        uri_to_path("s3://bucket/key.pdf")


@pytest.mark.parametrize("authority", ["localhost", "LOCALHOST"])
def test_uri_to_path_omits_localhost_authority(authority):
    assert uri_to_path(f"file://{authority}/tmp/a.md") == Path("/tmp/a.md")


@pytest.mark.parametrize(
    "uri",
    ["file://server/share/a.md", "file:////server/share/a.md"],
    ids=["authority", "empty_authority"],
)
def test_uri_to_path_reads_unc_host(uri):
    """Both spellings name a UNC host, the second through an empty authority."""
    path = uri_to_path(uri)
    assert path == Path("//server/share/a.md")
    assert PureWindowsPath(path) == PureWindowsPath(r"\\server\share\a.md")


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_uri_to_path_strips_windows_drive_prefix():
    assert uri_to_path("file:///C:/docs/a.pdf") == Path("C:\\docs\\a.pdf")
