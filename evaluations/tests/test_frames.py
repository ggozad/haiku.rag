import base64
import json
from pathlib import Path
from typing import cast

import httpx
import pytest

from evaluations.datasets.frames import (
    fetch_article_images,
    image_urls,
    inline_images,
)

HTML = """
<html><body>
  <img src="//thumb.wikimedia.org/a/logo.png" alt="logo">
  <img src="//thumb.wikimedia.org/a/logo.png" alt="same again">
  <img src="https://example.com/absolute.jpg">
  <img src="/w/relative.png">
  <img src="data:image/png;base64,AAAA">
  <img alt="no src at all">
</body></html>
"""


class StubResponse:
    def __init__(self, content=b"", status=200, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class StubClient:
    """Records every URL asked for and answers from a script."""

    def __init__(self, answers):
        self.answers = answers
        self.seen: list[str] = []

    def get(self, url, **kw):
        self.seen.append(url)
        return self.answers.get(url, StubResponse(status=404))


def test_image_urls_absolutises_protocol_relative_and_dedupes():
    """Parsoid emits `//host/path`, which docling cannot resolve without a
    base, so it never fetches and every picture lands without bytes."""
    assert image_urls(HTML) == [
        "https://thumb.wikimedia.org/a/logo.png",
        "https://example.com/absolute.jpg",
    ]


def test_image_urls_skips_relative_and_data_and_missing():
    """Only fetchable remote refs are returned."""
    urls = image_urls(HTML)
    assert not any(u.startswith("data:") for u in urls)
    assert not any(u.endswith("/w/relative.png") for u in urls)


def test_inline_images_rewrites_to_data_uris(tmp_path):
    png = tmp_path / "logo.png"
    png.write_bytes(b"\x89PNG-bytes")
    out = inline_images(HTML, {"https://thumb.wikimedia.org/a/logo.png": png})
    expected = base64.b64encode(b"\x89PNG-bytes").decode()
    assert f"data:image/png;base64,{expected}" in out
    # both occurrences of the same url are rewritten
    assert out.count("data:image/png;base64,") == 2 + 1  # two imgs + the literal one


def test_inline_images_leaves_uncached_refs_alone(tmp_path):
    """An image we could not cache keeps its original, unresolvable src, so the
    result degrades to a picture without bytes rather than a conversion-time
    fetch that could be rate-limited mid-corpus."""
    out = inline_images(HTML, {})
    assert "//thumb.wikimedia.org/a/logo.png" in out
    assert "https://example.com/absolute.jpg" in out


def test_fetch_article_images_caches_and_writes_a_marker(tmp_path):
    client = StubClient(
        {
            "https://thumb.wikimedia.org/a/logo.png": StubResponse(b"\x89PNG-1"),
            "https://example.com/absolute.jpg": StubResponse(b"\xff\xd8JPEG"),
        }
    )
    images = fetch_article_images(
        "https://en.wikipedia.org/wiki/X",
        HTML,
        tmp_path,
        cast(httpx.Client, client),
        throttle=0,
    )
    assert set(images) == {
        "https://thumb.wikimedia.org/a/logo.png",
        "https://example.com/absolute.jpg",
    }
    assert all(Path(p).read_bytes() for p in images.values())
    marker = tmp_path / "images" / "https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX.json"
    assert marker.exists()
    assert json.loads(marker.read_text())["failed"] == []


def test_fetch_article_images_serves_a_second_call_from_cache(tmp_path):
    client = StubClient(
        {
            "https://thumb.wikimedia.org/a/logo.png": StubResponse(b"\x89PNG-1"),
            "https://example.com/absolute.jpg": StubResponse(b"\xff\xd8JPEG"),
        }
    )
    uri = "https://en.wikipedia.org/wiki/X"
    fetch_article_images(uri, HTML, tmp_path, cast(httpx.Client, client), throttle=0)
    first = len(client.seen)
    again = fetch_article_images(
        uri, HTML, tmp_path, cast(httpx.Client, client), throttle=0
    )
    assert len(client.seen) == first, "a cached article must not refetch"
    assert set(again) == {
        "https://thumb.wikimedia.org/a/logo.png",
        "https://example.com/absolute.jpg",
    }


def test_fetch_article_images_records_failures_without_losing_the_article(tmp_path):
    """One 404 icon must not discard an article, but the loss has to be on the
    record: a silently partial corpus is the failure this dataset already had."""
    client = StubClient(
        {"https://thumb.wikimedia.org/a/logo.png": StubResponse(b"\x89PNG-1")}
    )
    images = fetch_article_images(
        "https://en.wikipedia.org/wiki/X",
        HTML,
        tmp_path,
        cast(httpx.Client, client),
        throttle=0,
        attempts=1,
    )
    assert set(images) == {"https://thumb.wikimedia.org/a/logo.png"}
    marker = json.loads(
        (
            tmp_path / "images" / "https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX.json"
        ).read_text()
    )
    assert marker["failed"] == ["https://example.com/absolute.jpg"]


def test_fetch_article_images_paces_requests(monkeypatch, tmp_path):
    """8 req/s is the measured ceiling from the build host; 16 draws 429s."""
    slept: list[float] = []
    monkeypatch.setattr("evaluations.datasets.frames.time.sleep", slept.append)
    client = StubClient(
        {
            "https://thumb.wikimedia.org/a/logo.png": StubResponse(b"a"),
            "https://example.com/absolute.jpg": StubResponse(b"b"),
        }
    )
    fetch_article_images(
        "https://en.wikipedia.org/wiki/X",
        HTML,
        tmp_path,
        cast(httpx.Client, client),
        throttle=0.125,
    )
    assert slept == [0.125, 0.125]


@pytest.mark.parametrize("size", [1, 2])
def test_image_urls_is_stable_across_calls(size):
    """Order is document order, so a cache key never depends on iteration luck."""
    assert image_urls(HTML) == image_urls(HTML)
