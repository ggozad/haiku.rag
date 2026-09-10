"""Expanding and enriching results through the database each came from."""

import pytest
from docling_core.types.doc.document import DoclingDocument
from pydantic_ai import ModelRetry

from haiku.rag.capabilities.rag import create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.store.exceptions import AmbiguousDatabaseError
from haiku.rag.store.models import Chunk, Document, DocumentItem, SearchResult
from tests.multi_db.helpers import (
    _config,
    _seed,
    _seed_expandable,
)


class TestExpansionRouting:
    @pytest.mark.asyncio
    async def test_expansion_routes_each_result_to_its_database(self, tmp_path):
        """A federating client has no repositories of its own, so expansion has
        to go through the database each result came from."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context(results)

        assert {r.source for r in expanded} == {"alpha", "beta"}
        for r in expanded:
            assert r.source is not None
            assert r.source in r.content

    @pytest.mark.asyncio
    async def test_an_expanded_result_keeps_its_source(self, tmp_path):
        """Expansion rebuilds the result, and the rebuilt one has to name the
        database it was expanded through."""
        config = _config(tmp_path, ["alpha"])
        await _seed_expandable(
            config, "alpha", ["cats sleep often", "cats also hunt", "cats purr"]
        )

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context(results)

        assert len(expanded) == 1
        assert "cats also hunt" in expanded[0].content, "expansion did not run"
        assert expanded[0].source == "alpha"

    @pytest.mark.asyncio
    async def test_a_federated_result_is_expanded_by_its_own_database(self, tmp_path):
        """Routing is not enough: each result has to come back carrying the
        neighbours of the database it was expanded through, and only those."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed_expandable(
            config, "alpha", ["cats sleep often", "alpha follows on"]
        )
        await _seed_expandable(config, "beta", ["cats also hunt", "beta follows on"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context(results)

        content = {r.source: r.content for r in expanded}
        assert "alpha follows on" in content["alpha"]
        assert "beta follows on" not in content["alpha"]
        assert "beta follows on" in content["beta"]

    @pytest.mark.asyncio
    async def test_expansion_keeps_tied_results_in_fused_order(self, tmp_path):
        """Fused scores tie often, so grouping by database must not reorder
        them: the tiebreak is the order they arrived in."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one about cats", "alpha two about cats"])
        await _seed(config, "beta", ["beta one about cats"])

        async with HaikuRAG(config=config) as rag:
            found = await rag.search("cats", search_type="fts", limit=10)
            by_source: dict[str, list[SearchResult]] = {}
            for result in found:
                by_source.setdefault(result.source or "", []).append(result)
            # Interleaved, so grouping by database is visible as a reordering.
            fused = [by_source["alpha"][0], by_source["beta"][0], by_source["alpha"][1]]
            for result in fused:
                result.score = 0.5

            expanded = await rag.expand_context(fused)

        assert [r.chunk_id for r in expanded] == [r.chunk_id for r in fused]


class TestPictureDeduplication:
    """One picture yields two chunks — a text-embedded one and an image-embedded
    one — that collapse to the best. Two databases holding the same picture are
    two results, not a duplicate."""

    @staticmethod
    def _picture(source, score):
        return SearchResult(
            content="a figure",
            score=score,
            source=source,
            chunk_id=f"{source}-c",
            document_id="doc-1",
            doc_item_refs=["#/pictures/0"],
        )

    def test_the_same_picture_in_two_databases_survives(self):
        from haiku.rag.client.search import _dedup_picture_chunks

        kept = _dedup_picture_chunks(
            [self._picture("alpha", 0.9), self._picture("clone", 0.5)]
        )

        assert [r.source for r in kept] == ["alpha", "clone"]

    def test_duplicates_within_one_database_still_collapse(self):
        from haiku.rag.client.search import _dedup_picture_chunks

        lower = self._picture("alpha", 0.5)
        higher = self._picture("alpha", 0.9)

        kept = _dedup_picture_chunks([lower, higher])

        assert kept == [higher]


class TestPictureRouting:
    @pytest.mark.asyncio
    async def test_a_picture_is_fetched_from_the_database_that_holds_it(self, tmp_path):
        """A `self_ref` repeats across databases, so the citation's source is
        what decides where the bytes come from."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [document] = await beta.document_repository.list_all(limit=1)
            assert document.id is not None
            await beta.document_item_repository.create_all(
                [
                    DocumentItem(
                        document_id=document.id,
                        self_ref="#/pictures/0",
                        position=99,
                        label="picture",
                        text="",
                        picture_data=b"beta-picture",
                    )
                ]
            )

            assert (
                await rag.get_picture_bytes(document.id, "#/pictures/0", "beta")
                == b"beta-picture"
            )
            assert (
                await rag.get_picture_bytes(document.id, "#/pictures/0", "alpha")
                is None
            )

    @pytest.mark.asyncio
    async def test_a_single_database_needs_no_source(self, temp_db_path):
        """One database is where the picture is, named or not."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            document = await rag.document_repository.create(
                Document(content="body", uri="test://one")
            )
            assert document.id is not None
            await rag.document_item_repository.create_all(
                [
                    DocumentItem(
                        document_id=document.id,
                        self_ref="#/pictures/0",
                        position=0,
                        label="picture",
                        text="",
                        picture_data=b"the-picture",
                    )
                ]
            )

            assert (
                await rag.get_picture_bytes(document.id, "#/pictures/0")
                == b"the-picture"
            )

    @pytest.mark.asyncio
    async def test_a_picture_lookup_without_a_source_is_refused(self, tmp_path):
        """Federating, nothing can say which database holds an unqualified
        reference."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(ValueError, match="source"):
                await rag.get_picture_bytes("doc-1", "#/pictures/0")


class TestFederatedEdges:
    @pytest.mark.asyncio
    async def test_expansion_passes_through_results_without_a_source(self, tmp_path):
        """A caller can hand `expand_context` results it built itself. Those name
        no database, so there is nowhere to expand them from."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        handmade = SearchResult(content="handmade", score=0.4, doc_item_refs=[])

        async with HaikuRAG(config=config) as rag:
            found = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context([*found, handmade])

        assert "handmade" in [r.content for r in expanded]
        scores = [r.score for r in expanded]
        assert scores == sorted(scores, reverse=True), "merged in score order"

    @pytest.mark.asyncio
    async def test_a_chunk_without_a_document_is_not_cited(self, tmp_path):
        """`Chunk.document_id` is optional, and a citation without a document has
        nothing to point at."""
        from unittest.mock import AsyncMock, patch

        from haiku.rag.capabilities.rag import RAGCapability
        from tests.capabilities.test_capabilities import (
            Deps,
            _single_database_client,
            make_context,
        )

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        orphan = _single_database_client()
        orphan.get_chunk_by_id.return_value = Chunk(
            id="orphan", document_id=None, content="no document"
        )

        capability = create_capability(config=config, defer_loading=False)
        run = await capability.for_run(make_context(Deps()))
        with patch.object(RAGCapability, "_ensure_rag", AsyncMock(return_value=orphan)):
            with pytest.raises(ModelRetry, match="None of the supplied chunk_ids"):
                await run._cite(["orphan"])


class TestVisualizationRouting:
    @staticmethod
    def _visualizable(name):
        """A one-page document whose only item has a bounding box to draw."""
        from docling_core.types.doc.base import BoundingBox, Size
        from docling_core.types.doc.document import ImageRef, ProvenanceItem
        from docling_core.types.doc.labels import DocItemLabel
        from PIL import Image as PilImageModule

        doc = DoclingDocument(name=name)
        doc.add_page(
            page_no=1,
            size=Size(width=612.0, height=792.0),
            image=ImageRef.from_pil(
                PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
            ),
        )
        doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text=f"{name} content",
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=50, t=700, r=550, b=650),
                charspan=(0, len(name) + 8),
            ),
        )
        return doc

    @pytest.mark.asyncio
    async def test_a_chunk_is_visualized_by_the_database_that_holds_it(self, tmp_path):
        """A chunk carries no database identity, so the source the caller holds
        is what reaches the pages and bounding boxes."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        dim = get_config().embeddings.model.vector_dim
        async with HaikuRAG(config=config, create=True, sources=["beta"]) as beta:
            document = await beta.import_document(
                self._visualizable("beta"),
                [
                    Chunk(
                        content="beta content",
                        embedding=[0.1] * dim,
                        order=0,
                        metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
                    )
                ],
                uri="test://beta/visualizable",
            )
            [chunk] = await beta.chunk_repository.get_by_document_id(document.id)

        async with HaikuRAG(config=config) as rag:
            assert len(await rag.visualize_chunk(chunk, source="beta")) == 1
            assert await rag.visualize_chunk(chunk, source="alpha") == []

    @pytest.mark.asyncio
    async def test_visualizing_without_a_source_is_refused(self, tmp_path):
        """Federating, nothing in a chunk says which database drew it."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(AmbiguousDatabaseError, match="source"):
                await rag.visualize_chunk(Chunk(content="x", document_id="doc-1"))
