"""FRAMES benchmark (google/frames-benchmark).

824 multi-hop questions, each grounded in two or more Wikipedia articles. The
corpus is the union of the articles linked per question, fetched from the
Wikipedia REST API at current revision and cached locally with the revision id
and fetch date.
"""

import ast
import base64
import hashlib
import json
import logging
import mimetypes
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import parse_qs, quote, unquote, urlsplit

import httpx
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from PIL import Image as PILImage
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import CitationMAPEvaluator, MAPEvaluator

logger = logging.getLogger(__name__)

USER_AGENT = "haiku.rag-evaluations (https://github.com/ggozad/haiku.rag)"
FETCH_ATTEMPTS = 3
THROTTLE_SECONDS = 1.0
RATE_LIMIT_BACKOFF_SECONDS = 60.0
THROTTLE_RETRIES = 20
# Measured on the corpus: one thumbnail a second yields 0.90 images a second,
# where 1.3 yields 0.58 and 2 yields 0.62. Asking for more earns 429s whose
# `Retry-After: 11` costs more than the extra requests return.
IMAGE_THROTTLE_SECONDS = 1.0


# Articles deleted from Wikipedia since FRAMES was authored; the questions
# linking them have lost their evidence and are excluded from the benchmark.
_DELETED_ARTICLES = frozenset(
    {
        "https://en.wikipedia.org/wiki/Nemanja_Marković",
        "https://en.wikipedia.org/wiki/Jack_Vance_(tennis)",
    }
)


def load_frames_test() -> Dataset:
    return load_dataset("google/frames-benchmark")["test"]


def question_is_answerable(doc: Mapping[str, Any]) -> bool:
    return not _DELETED_ARTICLES & set(question_expected_uris(doc))


def load_frames_questions() -> Dataset:
    """Answerable questions with a stable `id` (the dataset row number)."""
    dataset = load_frames_test().filter(question_is_answerable)
    return dataset.map(lambda row: {"id": str(row["Unnamed: 0"])})


def parse_wiki_links(raw: str) -> list[str]:
    """Extract URLs from a `wiki_links` value.

    The value is a Python-list-repr string. A single list element may pack
    several comma-separated URLs, and may carry trailing prose annotations;
    titles themselves can contain commas, so elements are split only where a
    new URL starts.
    """
    links: list[str] = []
    for element in ast.literal_eval(raw):
        for part in re.split(r",\s*(?=http)", element):
            tokens = part.split()
            if not tokens:
                continue
            url = tokens[0].strip(", ")
            if url:
                links.append(url)
    return links


def normalize_wiki_url(url: str) -> str | None:
    """Canonical article URL, used both as document uri and expected uri.

    Strips fragments, decodes percent-escapes, folds mobile hosts, resolves
    `index.php?title=` and `Special:Search` forms, and applies MediaWiki title
    canonicalization (underscores, first letter uppercased). Returns None for
    strings that don't point to an article.
    """
    url = url.strip()
    if not url:
        return None
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    host = parts.netloc.replace(".m.wikipedia.org", ".wikipedia.org")
    if host == "w.wiki":
        return url
    if parts.path.startswith("/wiki/"):
        title = parts.path[len("/wiki/") :]
    elif parts.path.startswith("/w/index.php"):
        query = parse_qs(parts.query)
        title = query.get("title", [""])[0]
        if not title or title.startswith("Special:"):
            title = query.get("search", [""])[0]
    else:
        return None
    title = unquote(title).replace(" ", "_").strip("_")
    if not title:
        return None
    return f"https://{host}/wiki/{title[0].upper() + title[1:]}"


def parse_revid(etag: str | None) -> str | None:
    """Revision id from a Wikipedia REST ETag header (`W/"<revid>/<uuid>"`)."""
    if not etag:
        return None
    match = re.search(r'"([^/"]+)/', etag)
    return match.group(1) if match else None


def strip_navigation(html: str) -> str:
    """Drop navigation chrome (navboxes, succession boxes) from parsoid HTML.

    These render as link-spam tables naming hundreds of related articles,
    polluting retrieval. Infoboxes carry no navigation role and are kept.
    """
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(attrs={"role": "navigation"}):
        element.decompose()
    return str(soup)


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "haiku.rag" / "evaluations" / "frames_articles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _fetch_category_page(
    host: str, title: str, client: httpx.Client
) -> tuple[str, str, str | None]:
    """Category pages render empty via page/html; synthesize a members list."""
    response = client.get(
        f"https://{host}/w/api.php",
        params={
            "action": "query",
            "list": "categorymembers",
            "cmtitle": title,
            "cmlimit": "500",
            "format": "json",
        },
    )
    response.raise_for_status()
    members = [m["title"] for m in response.json()["query"]["categorymembers"]]
    display = title.replace("_", " ")
    content = f"# {display}\n\nPages in this category:\n"
    content += "\n".join(f"- {member}" for member in members) + "\n"
    return content, "md", None


def _fetch_article_page(
    uri: str, client: httpx.Client
) -> tuple[str, str, str | None, str]:
    """Fetch parsoid HTML for an article; returns (content, format, revid, title)."""
    parts = urlsplit(uri)
    host = parts.netloc
    if host == "w.wiki":
        resolved = urlsplit(str(client.get(uri).url))
        host = resolved.netloc
        title = unquote(resolved.path[len("/wiki/") :])
    else:
        title = unquote(parts.path[len("/wiki/") :])
    response = client.get(
        f"https://{host}/api/rest_v1/page/html/{quote(title, safe='')}"
    )
    response.raise_for_status()
    revid = parse_revid(response.headers.get("etag"))
    return response.text, "html", revid, title


def _rate_limited(error: Exception) -> TypeGuard[httpx.HTTPStatusError]:
    return (
        isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429
    )


def _backoff_seconds(error: Exception, attempt: int) -> float:
    if _rate_limited(error):
        retry_after = error.response.headers.get("retry-after")
        return float(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SECONDS
    return 5.0 * attempt


def _fetch_with_retries[T](
    action: Callable[[], T],
    *,
    attempts: int = FETCH_ATTEMPTS,
    label: str = "",
) -> T:
    """Call `action`, retrying failures and waiting out rate limits.

    A 429 costs a wait rather than one of `attempts`, bounded by
    `THROTTLE_RETRIES`: a throttled host refuses for as long as it likes, and
    spending attempts on those refusals records a reachable resource as
    permanently unavailable.
    """
    attempt = 0
    waits = 0
    while True:
        try:
            return action()
        except Exception as e:
            if _rate_limited(e) and waits < THROTTLE_RETRIES:
                waits += 1
                time.sleep(_backoff_seconds(e, attempt + 1))
                continue
            attempt += 1
            if attempt >= attempts:
                raise
            logger.info(f"Retrying {label} after error: {e}")
            time.sleep(_backoff_seconds(e, attempt))


def _download_image(
    client: httpx.Client, url: str, target: Path, throttle: float
) -> None:
    time.sleep(throttle)
    response = client.get(url)
    response.raise_for_status()
    target.write_bytes(response.content)


def fetch_article(
    uri: str, cache_dir: Path, client: httpx.Client | None
) -> dict[str, Any] | None:
    """Return a corpus row for `uri`, fetching and caching it if needed.

    The cache holds the raw page plus a JSON sidecar with title, format,
    revision id, and fetch date; a present sidecar marks a complete entry and
    is served without network access.
    """
    base = quote(uri, safe="")
    meta_path = cache_dir / f"{base}.json"
    if meta_path.exists():
        row = json.loads(meta_path.read_text())
        row["path"] = str(cache_dir / f"{base}.{row['format']}")
        _cache_article_images(row, cache_dir, client)
        return row

    assert client is not None
    title = unquote(urlsplit(uri).path[len("/wiki/") :])
    # Wikimedia throttles sustained bot traffic; pace uncached fetches.
    time.sleep(THROTTLE_SECONDS)

    def _fetch_page() -> tuple[str, str, str | None, str]:
        if title.startswith("Category:"):
            content, format, revid = _fetch_category_page(
                urlsplit(uri).netloc, title, client
            )
            return content, format, revid, title
        return _fetch_article_page(uri, client)

    try:
        content, format, revid, title = _fetch_with_retries(_fetch_page, label=uri)
    except Exception as e:
        logger.warning(f"Failed to fetch {uri}: {e}")
        return None

    row: dict[str, Any] = {
        "uri": uri,
        "title": title.replace("_", " "),
        "format": format,
        "revid": revid,
        "fetched_at": datetime.now(UTC).date().isoformat(),
    }
    content_path = cache_dir / f"{base}.{format}"
    content_path.write_text(content)
    meta_path.write_text(json.dumps(row))
    row["path"] = str(content_path)
    _cache_article_images(row, cache_dir, client)
    return row


def _cache_article_images(
    row: dict[str, Any], cache_dir: Path, client: httpx.Client | None
) -> None:
    """Populate the image cache for an HTML article.

    Reads the navigation-stripped HTML, so the interface icons that
    `strip_navigation` discards are never downloaded.
    """
    if row["format"] != "html" or client is None:
        return
    html = Path(row["path"]).read_text()
    fetch_article_images(row["uri"], strip_navigation(html), cache_dir, client)


def image_urls(html: str) -> list[str]:
    """Fetchable image URLs referenced by `html`, in document order, deduped.

    Parsoid writes `//host/path`, which docling cannot resolve without a base.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for img in soup.find_all("img"):
        src = str(img.get("src") or "")
        if src.startswith("//"):
            src = f"https:{src}"
        elif not src.startswith(("http://", "https://")):
            continue
        if src not in seen:
            seen.add(src)
            urls.append(src)
    return urls


# Pillow refuses to write these as PNG, and `I` is deprecated for removal in
# Pillow 13. docling re-encodes every picture it extracts as PNG, and the
# refusal fails the whole document, not just the picture.
UNWRITABLE_PNG_MODES = frozenset({"CMYK", "YCbCr", "PA", "F", "I"})


def _png_writable(path: Path) -> tuple[bytes, str]:
    """The image's bytes and media type, recoded if Pillow cannot write it.

    Anything Pillow cannot open at all, SVG included, is passed through
    untouched.
    """
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        with PILImage.open(BytesIO(data)) as image:
            if image.mode not in UNWRITABLE_PNG_MODES:
                return data, mime
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue(), "image/png"
    except Exception as e:
        logger.info(f"Leaving {path.name} as it is: {e}")
        return data, mime


def inline_images(html: str, images: Mapping[str, Path | str]) -> str:
    """Rewrite each cached `<img src>` to a data: URI over its stored bytes.

    docling decodes inline data: URIs, so conversion reaches no network. An
    image with no cache entry keeps its unresolvable src.
    """
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        src = str(img.get("src") or "")
        key = f"https:{src}" if src.startswith("//") else src
        cached = images.get(key)
        if cached is None:
            continue
        data, mime = _png_writable(Path(cached))
        encoded = base64.b64encode(data).decode()
        img["src"] = f"data:{mime};base64,{encoded}"
    return str(soup)


def _image_path(images_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()
    suffix = Path(urlsplit(url).path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        suffix = ".png"
    return images_dir / f"{digest}{suffix}"


def fetch_article_images(
    uri: str,
    html: str,
    cache_dir: Path,
    client: httpx.Client | None,
    throttle: float = IMAGE_THROTTLE_SECONDS,
    attempts: int = FETCH_ATTEMPTS,
) -> dict[str, str]:
    """Cache every image `html` references; return url -> cached path.

    A marker sidecar per article records what resolved and what did not, so a
    resumed build refetches nothing. An unreachable image is recorded in
    `failed` rather than discarding the article.
    """
    images_dir = cache_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    marker_path = images_dir / f"{quote(uri, safe='')}.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        return {url: str(images_dir / name) for url, name in marker["images"].items()}

    if client is None:
        return {}
    images: dict[str, str] = {}
    failed: list[str] = []
    for url in image_urls(html):
        target = _image_path(images_dir, url)
        if target.exists():
            images[url] = target.name
            continue

        try:
            _fetch_with_retries(
                partial(_download_image, client, url, target, throttle),
                attempts=attempts,
                label=url,
            )
            images[url] = target.name
        except Exception as e:
            logger.info(f"Image unavailable for {uri}: {url}: {e}")
            failed.append(url)

    marker_path.write_text(
        json.dumps(
            {
                "images": images,
                "failed": failed,
                "fetched_at": datetime.now(UTC).date().isoformat(),
            }
        )
    )
    if failed:
        logger.warning(
            f"{len(failed)}/{len(images) + len(failed)} images unavailable for {uri}"
        )
    return {url: str(images_dir / name) for url, name in images.items()}


def question_expected_uris(doc: Mapping[str, Any]) -> tuple[str, ...]:
    uris: list[str] = []
    for link in parse_wiki_links(doc["wiki_links"]):
        normalized = normalize_wiki_url(link)
        if normalized is not None and normalized not in uris:
            uris.append(normalized)
    return tuple(uris)


_cached_corpus: list[dict[str, Any]] | None = None


def load_frames_corpus() -> list[dict[str, Any]]:
    """Fetch (or read from cache) every article linked by any question."""
    global _cached_corpus
    if _cached_corpus is None:
        uris: dict[str, None] = {}
        for doc in load_frames_questions():
            for uri in question_expected_uris(doc):
                uris.setdefault(uri)
        cache_dir = get_cache_dir()
        rows: list[dict[str, Any]] = []
        with httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60.0
        ) as client:
            for index, uri in enumerate(uris, start=1):
                row = fetch_article(uri, cache_dir, client)
                if row is not None:
                    rows.append(row)
                if index % 100 == 0:
                    logger.info(f"Fetched {index}/{len(uris)} articles")
        logger.info(f"Fetched {len(rows)}/{len(uris)} articles")
        if len(rows) < len(uris):
            raise RuntimeError(
                f"Fetched only {len(rows)}/{len(uris)} FRAMES articles; "
                "refusing to build a partial corpus. Re-run to resume from cache."
            )
        _cached_corpus = rows
    return _cached_corpus


def document_loader() -> Dataset:
    return Dataset.from_list(load_frames_corpus())


def map_frames_document(doc: Mapping[str, Any]) -> DocumentPayload:
    content = Path(doc["path"]).read_text()
    if doc["format"] == "html":
        content = strip_navigation(content)
        content = inline_images(
            content, fetch_article_images(doc["uri"], content, get_cache_dir(), None)
        )
    metadata: dict[str, str] = {"fetched_at": doc["fetched_at"]}
    if doc.get("revid"):
        metadata["revid"] = doc["revid"]
    return DocumentPayload(
        uri=doc["uri"],
        content=content,
        title=doc["title"],
        metadata=metadata,
        format=doc["format"],
    )


def map_frames_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    uris = question_expected_uris(doc)
    if not uris:
        return None
    return RetrievalSample(question=doc["Prompt"], expected_uris=uris)


def build_frames_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, str]]:
    return Case(
        name=f"{index}_{doc['id']}",
        inputs=doc["Prompt"],
        expected_output=doc["Answer"],
        metadata={
            "question_id": str(doc["id"]),
            "reasoning_types": str(doc["reasoning_types"]),
            "case_index": str(index),
        },
    )


FRAMES_SPEC = DatasetSpec(
    key="frames",
    db_filename="frames.lancedb",
    document_loader=document_loader,
    document_mapper=map_frames_document,
    qa_loader=load_frames_questions,
    qa_case_builder=build_frames_case,
    retrieval_loader=load_frames_questions,
    retrieval_mapper=map_frames_retrieval,
    retrieval_evaluators=[MAPEvaluator()],
    citation_evaluator=CitationMAPEvaluator(),
)
