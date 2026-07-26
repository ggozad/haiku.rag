"""FRAMES benchmark (google/frames-benchmark).

824 multi-hop questions, each grounded in two or more Wikipedia articles. The
corpus is the union of the articles linked per question, fetched from the
Wikipedia REST API at current revision and cached locally with the revision id
and fetch date.
"""

import ast
import json
import logging
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

import httpx
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import CitationMAPEvaluator, MAPEvaluator

logger = logging.getLogger(__name__)

USER_AGENT = "haiku.rag-evaluations (https://github.com/ggozad/haiku.rag)"
FETCH_ATTEMPTS = 3
THROTTLE_SECONDS = 1.0
RATE_LIMIT_BACKOFF_SECONDS = 60.0


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


def _backoff_seconds(error: Exception, attempt: int) -> float:
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
        retry_after = error.response.headers.get("retry-after")
        return float(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SECONDS
    return 5.0 * attempt


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
        return row

    assert client is not None
    title = unquote(urlsplit(uri).path[len("/wiki/") :])
    # Wikimedia throttles sustained bot traffic; pace uncached fetches.
    time.sleep(THROTTLE_SECONDS)
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            if title.startswith("Category:"):
                content, format, revid = _fetch_category_page(
                    urlsplit(uri).netloc, title, client
                )
            else:
                content, format, revid, title = _fetch_article_page(uri, client)
            break
        except Exception as e:
            if attempt == FETCH_ATTEMPTS:
                logger.warning(f"Failed to fetch {uri}: {e}")
                return None
            logger.info(f"Retrying {uri} after error: {e}")
            time.sleep(_backoff_seconds(e, attempt))

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
    return row


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
