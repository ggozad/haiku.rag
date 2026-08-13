"""Span coverage for the search pathway.

The pydantic-ai instrumentation already times the ``rag_search`` tool call as
a whole; these spans decompose that duration into embed / execute / hydrate /
rerank / expand / images so a slow search can be attributed to a stage, plus
the two one-off costs (store open, reranker weight load) that the run's first
tool call would otherwise absorb silently. The tests assert the span names and
the attributes queries are written against — renaming either breaks saved
Logfire views, so they are pinned here.

Spans are captured by monkeypatching each module's ``logfire`` object, the
same approach ``tests/ingester/test_workers.py`` uses for breaker events.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace

import pandas as pd
import pytest

from haiku.rag.capabilities import _base as capabilities_base
from haiku.rag.client import search as search_module
from haiku.rag.config import Config
from haiku.rag.reranking import base as reranking_base
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.store.repositories import chunk as chunk_repo_module
from haiku.rag.store.repositories.chunk import ChunkRepository


@dataclass
class RecordedSpan:
    name: str
    attributes: dict = field(default_factory=dict)

    def set_attribute(self, key, value):
        self.attributes[key] = value


class SpanRecorder:
    """Stand-in for the scoped ``logfire`` instance that records spans."""

    def __init__(self):
        self.spans: list[RecordedSpan] = []

    @contextmanager
    def span(self, name, **attributes):
        recorded = RecordedSpan(name, dict(attributes))
        self.spans.append(recorded)
        yield recorded

    def by_name(self, name: str) -> RecordedSpan:
        matches = [span for span in self.spans if span.name == name]
        assert matches, f"no {name!r} span; recorded {[s.name for s in self.spans]}"
        assert len(matches) == 1, f"expected one {name!r} span, got {len(matches)}"
        return matches[0]

    def names(self) -> list[str]:
        return [span.name for span in self.spans]


@pytest.fixture
def recorder(monkeypatch):
    """Record spans from every module on the search path."""
    recorder = SpanRecorder()
    for module in (
        reranking_base,
        chunk_repo_module,
        search_module,
        capabilities_base,
    ):
        monkeypatch.setattr(module, "logfire", recorder)
    return recorder


class StubReranker(RerankerBase):
    """Reranker that returns fixed scores, so the span is tested without a
    provider (local weights or remote API)."""

    _model = "stub-reranker"

    def __init__(self, scores: list[float]):
        self._scores = scores

    async def _rerank(self, query, chunks, top_n=10):
        return list(zip(chunks, self._scores))[:top_n]


def _chunks(count: int) -> list[Chunk]:
    return [Chunk(id=f"c{i}", content=f"chunk {i}") for i in range(count)]


async def test_rerank_span_records_fan_out_and_scores(recorder):
    reranker = StubReranker([0.9, 0.5, 0.2])

    await reranker.rerank("query", _chunks(3), top_n=2)

    span = recorder.by_name("search.rerank")
    assert span.attributes["provider"] == "StubReranker"
    assert span.attributes["model"] == "stub-reranker"
    # candidates is the limit*10 fan-out the search applied, the main lever
    # on rerank latency; top_n is what the caller asked for.
    assert span.attributes["candidates"] == 3
    assert span.attributes["top_n"] == 2
    assert span.attributes["results"] == 2
    assert span.attributes["top_score"] == 0.9
    assert span.attributes["score_spread"] == pytest.approx(0.4)


async def test_rerank_span_not_emitted_for_empty_candidates(recorder):
    assert await StubReranker([]).rerank("query", [], top_n=5) == []

    assert recorder.spans == []


async def test_rerank_span_tolerates_empty_results(recorder):
    """A reranker that drops every candidate still closes its span; the score
    attributes go None rather than raising on max() of an empty sequence."""
    await StubReranker([]).rerank("query", _chunks(2), top_n=2)

    span = recorder.by_name("search.rerank")
    assert span.attributes["results"] == 0
    assert span.attributes["top_score"] is None
    assert span.attributes["score_spread"] is None


class StubEmbedder:
    async def embed_query(self, text):
        return [0.1, 0.2, 0.3]


async def test_embed_span_records_provider_and_dim(recorder):
    store = SimpleNamespace(embedder=StubEmbedder(), _config=Config)
    repository = ChunkRepository(store)  # type: ignore[arg-type]

    assert await repository._embed_query("query") == [0.1, 0.2, 0.3]

    span = recorder.by_name("search.embed")
    assert span.attributes["provider"] == Config.embeddings.model.provider
    assert span.attributes["model"] == Config.embeddings.model.name
    assert span.attributes["dim"] == 3


class StubQueryResult:
    """LanceDB query builder stub: the awaited ``to_pandas`` is the point at
    which the real builder stops being lazy and the search actually runs."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def to_pandas(self):
        return self._df


class StubTableQuery:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def select(self, _columns):
        return self

    def where(self, _clause):
        return self

    async def to_list(self):
        return self._rows


async def test_execute_span_records_row_count(recorder):
    df = pd.DataFrame(
        [
            {
                "id": "c0",
                "document_id": "d0",
                "content": "hello",
                "metadata": "{}",
                "order": 0,
                "_relevance_score": 0.75,
            }
        ]
    )
    doc_rows = [{"id": "d0", "uri": "file://d0", "title": None, "metadata": "{}"}]
    store = SimpleNamespace(
        embedder=StubEmbedder(),
        _config=Config,
        ChunkRecord=_chunk_record_type(),
        document_meta_table=SimpleNamespace(query=lambda: StubTableQuery(doc_rows)),
    )
    repository = ChunkRepository(store)  # type: ignore[arg-type]

    results = await repository._process_search_results(StubQueryResult(df))  # type: ignore[arg-type]

    assert len(results) == 1
    # rows is the candidate count LanceDB returned, before rerank narrows it.
    assert recorder.by_name("search.execute").attributes["rows"] == 1
    # Turning that frame into Chunks is a separate stage: one more LanceDB
    # read for document metadata plus a per-row json.loads.
    hydrate = recorder.by_name("search.hydrate")
    assert hydrate.attributes["documents"] == 1
    assert hydrate.attributes["chunks"] == 1
    # Ordering matters for reading a trace: execution precedes hydration.
    assert recorder.names() == ["search.execute", "search.hydrate"]


def _chunk_record_type():
    """The record model Store builds per embedding dimension; only the fields
    _process_search_results populates matter here."""
    from pydantic import BaseModel

    class ChunkRecord(BaseModel):
        id: str
        document_id: str
        content: str
        content_fts: str = ""
        metadata: str = "{}"
        order: int = 0

    return ChunkRecord


async def test_expand_span_records_context_size(recorder):
    """Results without a document_id pass through unexpanded, so the span is
    exercised without a document_items table behind it."""
    client = SimpleNamespace(_config=Config, document_item_repository=None)
    results = [
        SearchResult(content="a" * 10, score=0.9),
        SearchResult(content="b" * 5, score=0.4),
    ]

    expanded = await search_module.expand_context(client, results)  # type: ignore[arg-type]

    assert len(expanded) == 2
    span = recorder.by_name("search.expand")
    assert span.attributes["documents"] == 1
    assert span.attributes["max_chars"] == Config.search.max_context_chars
    assert span.attributes["results_in"] == 2
    assert span.attributes["results_out"] == 2
    # context_chars is the payload about to be handed to the model, which ties
    # this stage's cost to the next model request's prompt size.
    assert span.attributes["context_chars"] == 15


class StubItemRepository:
    """Document-item repository serving one picture for one document."""

    def __init__(self, picture_ref: str, blob: bytes):
        self._picture_ref = picture_ref
        self._blob = blob

    async def get_caption_picture_refs(self, _document_id, _refs):
        return {}

    async def get_pictures_for_chunk(self, _document_id, _refs):
        return {self._picture_ref: self._blob}

    async def get_text_for_refs(self, _document_id, _refs):
        return {self._picture_ref: "a caption"}


async def test_images_span_records_documents_and_bytes(recorder):
    picture_ref = "#/pictures/0"
    blob = b"\x89PNG" + b"x" * 60
    client = SimpleNamespace(
        _config=Config,
        document_item_repository=StubItemRepository(picture_ref, blob),
    )
    results = [
        SearchResult(
            content="figure",
            score=0.9,
            document_id="d0",
            doc_item_refs=[picture_ref],
        )
    ]

    await search_module._populate_image_data(client, results)  # type: ignore[arg-type]

    assert results[0].image_data is not None
    span = recorder.by_name("search.images")
    # documents is the round-trip multiplier — up to three reads each.
    assert span.attributes["documents"] == 1
    assert span.attributes["pictures"] == 1
    # Raw blob bytes, not the ~4/3 larger base64 that gets attached.
    assert span.attributes["bytes"] == len(blob)


async def test_images_span_records_zero_when_nothing_attaches(recorder):
    """A result set with no pictures still opens the span, so the absence of
    image work is visible rather than inferred from a missing span."""
    client = SimpleNamespace(_config=Config, document_item_repository=None)

    await search_module._populate_image_data(
        client,  # type: ignore[arg-type]
        [SearchResult(content="text only", score=0.5)],
    )

    span = recorder.by_name("search.images")
    assert span.attributes["documents"] == 0
    assert span.attributes["pictures"] == 0
    assert span.attributes["bytes"] == 0


class StubRerankerClient:
    """Client whose ``reranker`` is a cached_property, like ``HaikuRAG``."""

    def __init__(self, config):
        self._config = config
        self.loads = 0

    @cached_property
    def reranker(self):
        self.loads += 1
        return StubReranker([1.0])


async def test_reranker_load_span_emitted_once_per_process(recorder):
    client = StubRerankerClient(Config)

    first = search_module._get_reranker(client)  # type: ignore[arg-type]
    second = search_module._get_reranker(client)  # type: ignore[arg-type]

    # The cached_property loaded weights exactly once; the span marks that
    # cold touch, so a search.reranker.load in a trace means a cold process.
    assert first is second
    assert client.loads == 1
    assert recorder.names() == ["search.reranker.load"]


class StubRag:
    """Stands in for HaikuRAG so _ensure_rag opens no real store."""

    opened = 0

    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        type(self).opened += 1
        return self

    async def __aexit__(self, *_exc):
        return False


@pytest.fixture
def capability(tmp_path, monkeypatch):
    from haiku.rag.capabilities.rag import RAGState, create_capability

    monkeypatch.setattr(capabilities_base, "HaikuRAG", StubRag)
    StubRag.opened = 0
    built = create_capability(db_path=tmp_path / "test.lancedb", config=Config)
    built.state = RAGState()
    return built


async def test_client_open_span_emitted_once_per_run(recorder, capability):
    first = await capability._ensure_rag()
    second = await capability._ensure_rag()

    # Store open is lazy and would otherwise be charged to whichever search
    # ran first; the double-checked lock means only one span per run.
    assert first is second
    assert StubRag.opened == 1
    span = recorder.by_name("rag.client.open")
    assert span.attributes["db_path"] == str(capability.db_path)


async def test_tool_search_span_records_position_in_run(
    recorder, capability, monkeypatch
):
    results = [SearchResult(content="evidence", score=0.9, chunk_id="c0")]

    async def _search_corpus(_rag, _query, limit=None, document_filter=None):
        return "formatted evidence", results

    monkeypatch.setattr(capabilities_base, "search_corpus", _search_corpus)

    await capability._search("first query", limit=3)
    await capability._search("second query", limit=3)

    searches = [s for s in recorder.spans if s.name == "ask.tool.search"]
    assert [s.attributes["search_index"] for s in searches] == [1, 2]
    assert searches[0].attributes["namespace"] == "rag"
    assert searches[0].attributes["max_searches"] == Config.qa.max_searches
    assert searches[0].attributes["limit"] == 3
    assert searches[0].attributes["results"] == 1
    assert searches[0].attributes["formatted_chars"] == len("formatted evidence")
    # Spans are recorded as they open, so this order is the nesting: the
    # one-off store open happens inside the FIRST search and never again,
    # which is what makes search #1 legitimately slower than the rest.
    assert recorder.names() == [
        "ask.tool.search",
        "rag.client.open",
        "ask.tool.search",
    ]


async def test_tool_search_span_not_emitted_when_budget_spent(
    recorder, capability, monkeypatch
):
    """A refused call emits no span: the ToolFailed is already on the tool
    span pydantic-ai opened, and search.* spans should mean work happened."""
    from pydantic_ai import ToolFailed

    async def _search_corpus(_rag, _query, limit=None, document_filter=None):
        return "", []

    monkeypatch.setattr(capabilities_base, "search_corpus", _search_corpus)
    capability.search_count = capability._max_searches

    with pytest.raises(ToolFailed):
        await capability._search("one too many", limit=3)

    assert "ask.tool.search" not in recorder.names()
