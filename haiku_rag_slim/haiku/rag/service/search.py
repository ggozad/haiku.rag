import asyncio
import json
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from haiku.rag.locations import LOCATIONS_KEY, item_locations
from haiku.rag.uri import is_local_uri, uri_to_path
from haiku.rag.utils.sql import escape_like_pattern, escape_sql_string

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models import Document, SearchResult

DocumentType = Literal[
    "pdf",
    "word",
    "powerpoint",
    "excel",
    "epub",
    "html",
    "markdown",
    "text",
    "code",
    "image",
    "other",
]

_TYPES_BY_EXTENSION: dict[str, DocumentType] = {
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
    ".odt": "word",
    ".pptx": "powerpoint",
    ".xlsx": "excel",
    ".csv": "excel",
    ".epub": "epub",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".md": "markdown",
    ".qmd": "markdown",
    ".rmd": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
    ".adoc": "text",
    ".asciidoc": "text",
    ".asc": "text",
    ".tex": "text",
    ".latex": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
}

# Types whose hits carry a line number: the file is the text that was indexed.
_LINE_TYPES = {"markdown", "text", "code"}
_MAX_LINE_FILE_BYTES = 4 * 1024 * 1024
_WORD = re.compile(r"\w{2,}")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or the to with".split()
)


class Box(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class PageLocation(BaseModel):
    page: int
    width: float
    height: float
    boxes: list[Box]


class Hit(BaseModel):
    rank: int
    score: float
    chunk_id: str | None
    document_id: str | None
    database: str | None
    folder_id: str | None
    title: str | None
    path: str | None
    uri: str | None
    document_type: DocumentType
    content_type: str | None
    page_numbers: list[int]
    line: int | None
    headings: list[str] | None
    labels: list[str]
    snippet: str
    snippet_highlights: list[tuple[int, int]]
    text: str
    pages: list[PageLocation]
    context: str | None = None


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    hits: list[Hit]


def document_type(uri: str | None, content_type: str | None) -> DocumentType:
    """What kind of file a document is, from its URI's extension."""
    from haiku.rag.converters.text_utils import TextFileHandler

    suffix = Path(uri.split("#", 1)[0]).suffix.lower() if uri else ""
    if suffix in _TYPES_BY_EXTENSION:
        return _TYPES_BY_EXTENSION[suffix]
    if suffix in TextFileHandler.text_extensions:
        return "code"
    if content_type == "application/pdf":
        return "pdf"
    return "other"


def local_path(uri: str | None) -> Path | None:
    if uri is None or not is_local_uri(uri) or "#" in uri:
        return None
    return uri_to_path(uri)


def query_terms(query: str) -> list[str]:
    """The query's words worth highlighting, lowercased, first seen first."""
    terms: list[str] = []
    for word in _WORD.findall(query.lower()):
        if word not in _STOPWORDS and word not in terms:
            terms.append(word)
    return terms


def snippet(
    text: str, terms: list[str], width: int
) -> tuple[str, list[tuple[int, int]]]:
    """At most `width` characters of `text` around the first term found, and
    where each term occurs in them."""
    flat = " ".join(text.split())
    lowered = flat.lower()
    found = [i for term in terms if (i := lowered.find(term)) >= 0]
    start = 0
    if len(flat) > width and found:
        start = max(0, min(found) - width // 4)
        start = min(start, len(flat) - width)
        # Back to the start of the word it cuts.
        start = flat.rfind(" ", 0, start) + 1
    window = flat[start : start + width]
    if start + width < len(flat) and (space := window.rfind(" ")) > width // 2:
        window = window[:space]
    prefix = "..." if start > 0 else ""
    suffix = "..." if start + len(window) < len(flat) else ""
    shown = f"{prefix}{window}{suffix}"
    highlights = []
    if terms:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE
        )
        highlights = [(m.start(), m.end()) for m in pattern.finditer(shown)]
    return shown, highlights


def find_line(path: Path, text: str) -> int | None:
    """The 1-based line of `path` where `text` starts, by its first line long
    enough to place it."""
    try:
        if path.stat().st_size > _MAX_LINE_FILE_BYTES:
            return None
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for candidate in text.splitlines():
        probe = candidate.strip().lstrip("#>-*`|0123456789. ").strip()
        if len(probe) < 4:
            continue
        probe = probe[:80]
        for number, line in enumerate(lines, 1):
            if probe in line:
                return number
    return None


def metadata_filter(key: str, value: str) -> str:
    """SQL filter on a key of the documents' JSON metadata."""
    fragment = escape_sql_string(escape_like_pattern(json.dumps(value)[1:-1]))
    return f"metadata LIKE '%\"{key}\": \"{fragment}\"%' ESCAPE '\\'"


def id_filter(document_id: str) -> str:
    return f"id = '{escape_sql_string(document_id)}'"


class SearchService:
    """Searches the configured databases through one read-only client.

    The client opens on the first search and closes after `search_idle_s`
    without one, which releases the embedding model it holds.
    """

    def __init__(self, config: "AppConfig"):
        self._config = config
        self._lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._client: HaikuRAG | None = None
        self._opened_for: AppConfig | None = None
        self._active = 0
        self._timer: asyncio.TimerHandle | None = None
        self._docling: OrderedDict[tuple[str | None, str], DoclingDocument] = (
            OrderedDict()
        )

    @property
    def config(self) -> "AppConfig":
        return self._config

    def reconfigure(self, config: "AppConfig") -> None:
        """Search the databases `config` places once no search is running."""
        self._config = config

    @property
    def loaded(self) -> bool:
        return self._client is not None

    @asynccontextmanager
    async def client(self) -> AsyncIterator["HaikuRAG"]:
        """The open client, held open while the caller uses it."""
        async with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._opened_for is not self._config and self._active == 0:
                await self._close()
            if self._client is None:
                await self._open()
            self._active += 1
            assert self._client is not None
            client = self._client
        try:
            yield client
        finally:
            self._active -= 1
            if self._active == 0:
                loop = asyncio.get_running_loop()
                self._timer = loop.call_later(
                    self._config.ingester.service.search_idle_s, self._idle
                )

    async def _open(self) -> None:
        from haiku.rag.client import HaikuRAG
        from haiku.rag.client.scope import DatabaseScope

        stack = AsyncExitStack()
        scope = DatabaseScope.resolve(self._config)
        self._client = await stack.enter_async_context(
            HaikuRAG._covering(scope, self._config, read_only=True)
        )
        self._stack = stack
        self._opened_for = self._config

    def _idle(self) -> None:
        self._timer = None
        asyncio.ensure_future(self._close_if_idle())

    async def _close_if_idle(self) -> None:
        async with self._lock:
            if self._active == 0:
                await self._close()

    async def _close(self) -> None:
        stack, self._stack, self._client = self._stack, None, None
        self._opened_for = None
        self._docling.clear()
        if stack is not None:
            await stack.aclose()

    async def while_idle(self, work: Callable[[], Awaitable[object]]) -> bool:
        """Run `work` with searches held back, unless one is running.

        The client is closed first, so the next search reads the databases
        as `work` left them.
        """
        async with self._lock:
            if self._active:
                return False
            await self._close()
            await work()
            return True

    async def aclose(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        async with self._lock:
            await self._close()

    async def search(
        self,
        query: str,
        *,
        limit: int,
        search_type: Literal["hybrid", "vector", "fts"] = "hybrid",
        databases: list[str] | None = None,
        folders: list[str] | None = None,
        document_id: str | None = None,
        context: bool = False,
    ) -> SearchResponse:
        started = time.perf_counter()
        clauses = []
        if document_id is not None:
            clauses.append(id_filter(document_id))
        if folders:
            clauses.append(
                "("
                + " OR ".join(metadata_filter("source_id", f) for f in folders)
                + ")"
            )
        async with self.client() as rag:
            results = await rag.search(
                query,
                limit=limit,
                search_type=search_type,
                filter=" AND ".join(clauses) or None,
                include_images=False,
                sources=databases,
            )
            expanded = await rag.expand_context(results) if context else None
            terms = query_terms(query)
            hits = [
                await self._hit(rag, rank, result, terms)
                for rank, result in enumerate(results, 1)
            ]
        if expanded is not None:
            contexts = {
                (wider.source, chunk_id): wider.content
                for wider in expanded
                for chunk_id in wider.chunk_ids or [wider.chunk_id]
            }
            for hit in hits:
                hit.context = contexts.get((hit.database, hit.chunk_id))
        return SearchResponse(
            query=query,
            took_ms=round((time.perf_counter() - started) * 1000, 1),
            hits=hits,
        )

    async def _hit(
        self, rag: "HaikuRAG", rank: int, result: "SearchResult", terms: list[str]
    ) -> Hit:
        meta = result.document_meta or {}
        content_type = meta.get("content_type")
        kind = document_type(result.document_uri, content_type)
        path = local_path(result.document_uri)
        line = None
        if path is not None and kind in _LINE_TYPES:
            line = await asyncio.to_thread(find_line, path, result.content)
        shown, highlights = snippet(
            result.content, terms, self._config.ingester.service.snippet_chars
        )
        locations = result.chunk_meta.get(LOCATIONS_KEY)
        if locations is None and result.page_numbers:
            locations = await self._locate(rag, result)
        return Hit(
            rank=rank,
            score=result.score,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            database=result.source,
            folder_id=meta.get("source_id"),
            title=result.document_title or (path.name if path is not None else None),
            path=str(path) if path is not None else None,
            uri=result.document_uri,
            document_type=kind,
            content_type=content_type,
            page_numbers=result.page_numbers,
            line=line,
            headings=result.headings,
            labels=result.labels,
            snippet=shown,
            snippet_highlights=highlights,
            text=result.content,
            pages=[_page(location) for location in locations or []],
        )

    async def _locate(
        self, rag: "HaikuRAG", result: "SearchResult"
    ) -> list[dict[str, Any]]:
        """Boxes for a chunk stored before chunks recorded them, from the
        stored document. The last few documents read are kept."""
        assert result.document_id is not None
        key = (result.source, result.document_id)
        document = self._docling.get(key)
        if document is None:
            document = await self._stored_docling(
                rag, result.source, result.document_id
            )
            if document is None:
                return []
            self._docling[key] = document
            while len(self._docling) > 4:
                self._docling.popitem(last=False)
        self._docling.move_to_end(key)
        return await asyncio.to_thread(item_locations, document, result.doc_item_refs)

    async def _stored_docling(
        self, rag: "HaikuRAG", source: str | None, document_id: str
    ) -> "DoclingDocument | None":
        owner = await rag.reader_for(source)
        assert owner is not None, "a result names its database"
        repository = owner.document_repository
        stored = await repository.get_docling_data(document_id)
        if stored is None:
            return None
        pages = await repository.get_pages_data(document_id)
        blob = pages.docling_pages if pages is not None else None
        return await asyncio.to_thread(_with_pages, stored, blob)


def _with_pages(stored: "Document", pages: bytes | None) -> "DoclingDocument | None":
    """The stored document with its pages' sizes, which are stored apart from
    its structure, and without their images."""
    from docling_core.types.doc.document import PageItem

    from haiku.rag.store.compression import decompress_json

    document = stored.get_docling_document()
    if document is None or pages is None:
        return document
    for number, page in json.loads(decompress_json(pages)).items():
        page.pop("image", None)
        document.pages[int(number)] = PageItem.model_validate(page)
    return document


def _page(location: dict[str, Any]) -> PageLocation:
    return PageLocation(
        page=location["page"],
        width=location["width"],
        height=location["height"],
        boxes=[
            Box(left=left, top=top, right=right, bottom=bottom)
            for left, top, right, bottom in location["boxes"]
        ],
    )
