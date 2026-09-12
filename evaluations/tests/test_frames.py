import base64
import io
import json
from pathlib import Path
from typing import cast

import httpx
import pytest
from PIL import Image

from evaluations.datasets.frames import (
    _fetch_with_retries,
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


def throttled(url: str, retry_after: str = "0") -> httpx.Response:
    return httpx.Response(
        429,
        headers={"retry-after": retry_after},
        request=httpx.Request("GET", url),
    )


class ThrottlingClient:
    """Answers 429 a fixed number of times before serving the image."""

    def __init__(self, url: str, refusals: int, content: bytes = b"\x89PNG"):
        self.url = url
        self.refusals = refusals
        self.content = content
        self.seen: list[str] = []

    def get(self, url, **kw):
        self.seen.append(url)
        if url != self.url:
            return StubResponse(status=404)
        if self.refusals > 0:
            self.refusals -= 1
            return throttled(url)
        return httpx.Response(
            200, content=self.content, request=httpx.Request("GET", url)
        )


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
    """Every request waits the configured throttle, 429 or not."""
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


def test_fetch_article_images_waits_out_a_rate_limit(monkeypatch, tmp_path):
    """A 429 is the host asking us to wait, so it must not spend an attempt:
    under sustained throttling every attempt goes on `Retry-After` sleeps and a
    reachable image lands in `failed` for good."""
    slept: list[float] = []
    monkeypatch.setattr("evaluations.datasets.frames.time.sleep", slept.append)
    url = "https://thumb.wikimedia.org/a/logo.png"
    client = ThrottlingClient(url, refusals=5)
    images = fetch_article_images(
        "https://en.wikipedia.org/wiki/X",
        HTML,
        tmp_path,
        cast(httpx.Client, client),
        throttle=0,
        attempts=3,
    )
    assert url in images
    assert Path(images[url]).read_bytes() == b"\x89PNG"
    marker = json.loads(
        (
            tmp_path / "images" / "https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX.json"
        ).read_text()
    )
    assert url not in marker["failed"]


def test_fetch_article_images_stops_waiting_on_a_permanent_rate_limit(
    monkeypatch, tmp_path
):
    """Waiting is bounded, so a host that never lets up still terminates."""
    monkeypatch.setattr("evaluations.datasets.frames.time.sleep", lambda _: None)
    monkeypatch.setattr("evaluations.datasets.frames.THROTTLE_RETRIES", 2)
    url = "https://thumb.wikimedia.org/a/logo.png"
    client = ThrottlingClient(url, refusals=1000)
    images = fetch_article_images(
        "https://en.wikipedia.org/wiki/X",
        HTML,
        tmp_path,
        cast(httpx.Client, client),
        throttle=0,
        attempts=2,
    )
    assert url not in images
    marker = json.loads(
        (
            tmp_path / "images" / "https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX.json"
        ).read_text()
    )
    assert url in marker["failed"]
    assert client.seen.count(url) < 20, "waiting must be bounded"


def test_fetch_with_retries_spends_attempts_only_on_real_errors(monkeypatch):
    monkeypatch.setattr("evaluations.datasets.frames.time.sleep", lambda _: None)
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        _fetch_with_retries(action, attempts=3)
    assert calls["n"] == 3


def test_fetch_with_retries_returns_the_first_success(monkeypatch):
    monkeypatch.setattr("evaluations.datasets.frames.time.sleep", lambda _: None)
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("GET", "https://x/y.png"),
                response=throttled("https://x/y.png"),
            )
        return "done"

    assert _fetch_with_retries(action, attempts=2) == "done"
    assert calls["n"] == 3


@pytest.mark.parametrize(
    ("mode", "format"), [("CMYK", "JPEG"), ("YCbCr", "JPEG"), ("F", "TIFF")]
)
def test_inline_images_recodes_what_pillow_cannot_write_as_png(mode, format, tmp_path):
    """docling re-encodes every picture as PNG, so one image in a mode Pillow
    cannot write fails the whole document and takes the corpus build with it."""
    source = tmp_path / f"image.{format.lower()}"
    Image.new(mode, (80, 80)).save(source, format=format)

    out = inline_images(HTML, {"https://thumb.wikimedia.org/a/logo.png": source})

    encoded = out.split("base64,")[1].split('"')[0]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
        assert im.mode == "RGB"
        assert im.size == (80, 80)


def test_inline_images_leaves_a_writable_image_untouched(tmp_path):
    png = tmp_path / "rgb.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(png, format="PNG")
    raw = png.read_bytes()

    out = inline_images(HTML, {"https://thumb.wikimedia.org/a/logo.png": png})

    encoded = out.split("base64,")[1].split('"')[0]
    assert base64.b64decode(encoded) == raw


def test_inline_images_passes_through_what_pillow_cannot_open(tmp_path):
    """SVG is a normal Wikimedia image and is not a raster Pillow can parse."""
    svg = tmp_path / "icon.svg"
    svg.write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>")

    out = inline_images(HTML, {"https://thumb.wikimedia.org/a/logo.png": svg})

    encoded = out.split("base64,")[1].split('"')[0]
    assert base64.b64decode(encoded) == svg.read_bytes()
