import json
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    import pandas as pd
    from lancedb.query import AsyncQueryBase

from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from haiku.rag.store.engine import Store, query_to_pydantic
from haiku.rag.store.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Repository for Chunk operations."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.embedder = store.embedder

    async def _ensure_fts_index(self) -> None:
        """Ensure FTS index exists on the content_fts column."""
        try:
            await self.store.chunks_table.create_index(
                "content_fts",
                config=FTS(with_position=True, remove_stop_words=False),
                replace=True,
            )
        except Exception as e:
            # Log the error but don't fail - FTS might already exist
            logger.debug(f"FTS index creation skipped: {e}")

    def _contextualize_content(self, chunk: Chunk) -> str:
        """Generate contextualized content for FTS by prepending headings."""
        meta = chunk.get_chunk_metadata()
        if meta.headings:
            return "\n".join(meta.headings) + "\n" + chunk.content
        return chunk.content

    async def create(self, entity: Chunk | list[Chunk]) -> Chunk | list[Chunk]:
        """Create one or more chunks in the database.

        Chunks must have embeddings set before calling this method.
        Use client._ensure_chunks_embedded() to embed chunks if needed.
        """
        self.store._assert_writable()
        # Handle single chunk
        if isinstance(entity, Chunk):
            assert entity.document_id, "Chunk must have a document_id to be created"
            assert entity.embedding is not None, "Chunk must have an embedding"

            chunk_id = str(uuid4())

            chunk_record = self.store.ChunkRecord(
                id=chunk_id,
                document_id=entity.document_id,
                content=entity.content,
                content_fts=self._contextualize_content(entity),
                metadata=json.dumps(
                    {k: v for k, v in entity.metadata.items() if k != "order"}
                ),
                order=int(entity.order),
                vector=entity.embedding,
            )

            await self.store.chunks_table.add([chunk_record])

            entity.id = chunk_id
            return entity

        # Handle batch of chunks
        chunks = entity
        if not chunks:
            return []

        # Validate all chunks have document_id and embedding
        for chunk in chunks:
            assert chunk.document_id, "All chunks must have a document_id to be created"
            assert chunk.embedding is not None, "All chunks must have embeddings"

        # Prepare all chunk records
        chunk_records = []
        for chunk in chunks:
            chunk_id = str(uuid4())

            assert chunk.document_id is not None
            assert chunk.embedding is not None
            chunk_record = self.store.ChunkRecord(
                id=chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                content_fts=self._contextualize_content(chunk),
                metadata=json.dumps(
                    {k: v for k, v in chunk.metadata.items() if k != "order"}
                ),
                order=int(chunk.order),
                vector=chunk.embedding,
            )
            chunk_records.append(chunk_record)
            chunk.id = chunk_id

        # Single batch insert for all chunks
        await self.store.chunks_table.add(chunk_records)

        return chunks

    async def get_by_id(self, entity_id: str) -> Chunk | None:
        """Get a chunk by its ID."""
        results = await query_to_pydantic(
            self.store.chunks_table.query().where(f"id = '{entity_id}'").limit(1),
            self.store.ChunkRecord,
        )

        if not results:
            return None

        chunk_record = results[0]
        md = json.loads(chunk_record.metadata)
        return Chunk(
            id=chunk_record.id,
            document_id=chunk_record.document_id,
            content=chunk_record.content,
            metadata=md,
            order=chunk_record.order,
        )

    async def update(self, entity: Chunk) -> Chunk:
        """Update an existing chunk.

        Chunk must have embedding set before calling this method.
        """
        self.store._assert_writable()
        assert entity.id, "Chunk ID is required for update"
        assert entity.embedding is not None, "Chunk must have an embedding"

        await self.store.chunks_table.update(
            {
                "document_id": entity.document_id,
                "content": entity.content,
                "content_fts": self._contextualize_content(entity),
                "metadata": json.dumps(
                    {k: v for k, v in entity.metadata.items() if k != "order"}
                ),
                "order": int(entity.order),
                "vector": entity.embedding,
            },
            where=f"id = '{entity.id}'",
        )
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete a chunk by its ID."""
        self.store._assert_writable()
        chunk = await self.get_by_id(entity_id)
        if chunk is None:
            return False

        await self.store.chunks_table.delete(f"id = '{entity_id}'")
        return True

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Chunk]:
        """List all chunks with optional pagination."""
        query = self.store.chunks_table.query()

        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        results = await query_to_pydantic(query, self.store.ChunkRecord)

        chunks: list[Chunk] = []
        for rec in results:
            md = json.loads(rec.metadata)
            chunks.append(
                Chunk(
                    id=rec.id,
                    document_id=rec.document_id,
                    content=rec.content,
                    metadata=md,
                    order=rec.order,
                )
            )
        return chunks

    async def delete_all(self) -> None:
        """Delete all chunks from the database."""
        self.store._assert_writable()
        # Drop and recreate table to clear all data
        await self.store.db.drop_table("chunks")
        self.store.chunks_table = await self.store.db.create_table(
            "chunks", schema=self.store.ChunkRecord
        )
        # Create FTS index on content_fts (contextualized content) for better search
        await self.store.chunks_table.create_index(
            "content_fts",
            config=FTS(with_position=True, remove_stop_words=False),
            replace=True,
        )

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        self.store._assert_writable()
        chunks = await self.get_by_document_id(document_id)

        if not chunks:
            return False

        await self.store.chunks_table.delete(f"document_id = '{document_id}'")
        return True

    async def search(
        self,
        query: str,
        limit: int = 5,
        search_type: str = "hybrid",
        filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for relevant chunks using the specified search method.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            search_type: Type of search - "vector", "fts", or "hybrid" (default).
            filter: Optional SQL WHERE clause to filter documents before searching chunks.

        Returns:
            List of (chunk, score) tuples ordered by relevance.
        """
        if not query.strip():
            return []

        chunk_filter: str | None = None
        if filter:
            # Translate the document-level filter into a chunk-level
            # document_id IN (...) clause so LanceDB can combine it with
            # limit. The previous two-step pattern (materialize top-N,
            # filter in pandas, head(limit)) silently under-returned
            # whenever the top-N window lacked `limit` matching chunks.
            docs_df = await (
                self.store.documents_table.query()
                .select(["id"])
                .where(filter)
                .to_pandas()
            )
            if docs_df.empty:
                return []
            id_list = ", ".join(f"'{d}'" for d in docs_df["id"])
            chunk_filter = f"document_id IN ({id_list})"

        if search_type == "vector":
            query_embedding = await self.embedder.embed_query(query)
            results = (
                self.store.chunks_table.query()
                .nearest_to(query_embedding)
                .column("vector")
                .refine_factor(self.store._config.search.vector_refine_factor)
            )
        elif search_type == "fts":
            results = self.store.chunks_table.query().nearest_to_text(
                query, columns="content_fts"
            )
        else:  # hybrid (default)
            query_embedding = await self.embedder.embed_query(query)
            reranker = RRFReranker()
            results = (
                self.store.chunks_table.query()
                .nearest_to(query_embedding)
                .column("vector")
                .nearest_to_text(query, columns="content_fts")
                .refine_factor(self.store._config.search.vector_refine_factor)
                .rerank(reranker)
            )

        if chunk_filter is not None:
            results = results.where(chunk_filter)
        results = results.limit(limit)
        return await self._process_search_results(results)

    async def get_by_document_id(
        self,
        document_id: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Chunk]:
        """Get chunks for a specific document with optional pagination.

        Args:
            document_id: The document ID to get chunks for.
            limit: Maximum number of chunks to return. None for all.
            offset: Number of chunks to skip. None for no offset.

        Returns:
            List of chunks ordered by their order field.
        """
        query = self.store.chunks_table.query().where(f"document_id = '{document_id}'")

        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        results = await query_to_pydantic(query, self.store.ChunkRecord)

        # Get document info (only metadata columns, skip content/docling blobs)
        doc_rows = await (
            self.store.documents_table.query()
            .select(["id", "uri", "title", "metadata"])
            .where(f"id = '{document_id}'")
            .limit(1)
            .to_list()
        )

        doc_uri = doc_rows[0]["uri"] if doc_rows else None
        doc_title = doc_rows[0]["title"] if doc_rows else None
        doc_meta = doc_rows[0].get("metadata", "{}") if doc_rows else "{}"

        chunks: list[Chunk] = []
        for rec in results:
            md = json.loads(rec.metadata)
            chunks.append(
                Chunk(
                    id=rec.id,
                    document_id=rec.document_id,
                    content=rec.content,
                    metadata=md,
                    order=rec.order,
                    document_uri=doc_uri,
                    document_title=doc_title,
                    document_meta=json.loads(doc_meta),
                )
            )

        chunks.sort(key=lambda c: c.order)
        return chunks

    async def count_by_document_id(self, document_id: str) -> int:
        """Count the number of chunks for a specific document."""
        df = await (
            self.store.chunks_table.query()
            .select(["id"])
            .where(f"document_id = '{document_id}'")
            .to_pandas()
        )
        return len(df)

    async def get_chunks_in_range(
        self, document_id: str, min_order: int, max_order: int
    ) -> list[Chunk]:
        """Get chunks for a document within an order range.

        Args:
            document_id: The document ID to get chunks for.
            min_order: Minimum order value (inclusive).
            max_order: Maximum order value (inclusive).

        Returns:
            List of chunks within the order range.
        """
        where = (
            f"document_id = '{document_id}'"
            f" AND `order` >= {min_order}"
            f" AND `order` <= {max_order}"
        )
        results = await query_to_pydantic(
            self.store.chunks_table.query().where(where), self.store.ChunkRecord
        )
        return [
            Chunk(
                id=rec.id,
                document_id=rec.document_id,
                content=rec.content,
                metadata=json.loads(rec.metadata),
                order=rec.order,
            )
            for rec in results
        ]

    async def _process_search_results(
        self, query_result: "pd.DataFrame | AsyncQueryBase"
    ) -> list[tuple[Chunk, float]]:
        """Process search results into chunks with document info and scores.

        Args:
            query_result: Either a pandas DataFrame or a LanceDB async query result
        """
        import pandas as pd

        def extract_scores(df: pd.DataFrame) -> list[float]:
            """Extract scores from DataFrame columns based on search type."""
            if "_distance" in df.columns:
                # Vector search - convert distance to similarity
                return ((df["_distance"] + 1).rdiv(1)).clip(lower=0.0).tolist()
            elif "_relevance_score" in df.columns:
                # Hybrid search - relevance score (higher is better)
                return df["_relevance_score"].tolist()
            elif "_score" in df.columns:
                # FTS search - score (higher is better)
                return df["_score"].tolist()
            else:
                raise ValueError("Unknown search result format, cannot extract scores")

        # Convert everything to DataFrame for uniform processing
        if isinstance(query_result, pd.DataFrame):
            df = query_result
        else:
            # Convert LanceDB query result to DataFrame
            df = await query_result.to_pandas()

        # Extract scores
        scores = extract_scores(df)

        # Convert DataFrame rows to ChunkRecords
        pydantic_results = [
            self.store.ChunkRecord(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                content=str(row["content"]),
                content_fts=str(row.get("content_fts", "")),
                metadata=str(row["metadata"]),
                order=int(row["order"]) if "order" in row else 0,
            )
            for _, row in df.iterrows()
        ]

        # Collect all unique document IDs for batch lookup
        document_ids = list(set(chunk.document_id for chunk in pydantic_results))

        # Batch fetch document metadata (skip content/docling blobs)
        documents_map: dict[str, dict] = {}
        if document_ids:
            id_list = "', '".join(document_ids)
            where_clause = f"id IN ('{id_list}')"
            doc_rows = await (
                self.store.documents_table.query()
                .select(["id", "uri", "title", "metadata"])
                .where(where_clause)
                .to_list()
            )
            documents_map = {str(row["id"]): row for row in doc_rows}

        # Build final results with document info
        chunks_with_scores = []
        for i, chunk_record in enumerate(pydantic_results):
            doc = documents_map.get(chunk_record.document_id)
            chunk = Chunk(
                id=chunk_record.id,
                document_id=chunk_record.document_id,
                content=chunk_record.content,
                metadata=json.loads(chunk_record.metadata),
                order=chunk_record.order,
                document_uri=doc["uri"] if doc else None,
                document_title=doc["title"] if doc else None,
                document_meta=json.loads(doc.get("metadata", "{}") if doc else "{}"),
            )
            score = scores[i] if i < len(scores) else 1.0
            chunks_with_scores.append((chunk, score))

        return chunks_with_scores
