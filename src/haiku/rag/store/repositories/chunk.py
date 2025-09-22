import asyncio
import inspect
import json
import logging
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument
from lancedb.rerankers import RRFReranker

from haiku.rag.chunker import chunker
from haiku.rag.config import Config
from haiku.rag.embeddings import get_embedder
from haiku.rag.store.engine import DocumentRecord, Store
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.utils import load_callable, text_to_docling_document

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Repository for Chunk operations."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.embedder = get_embedder()
        self._optimize_lock = asyncio.Lock()

    def _ensure_fts_index(self) -> None:
        """Ensure FTS index exists on the content column."""
        try:
            self.store.chunks_table.create_fts_index(
                "content", replace=True, with_position=True, remove_stop_words=False
            )
        except Exception as e:
            # Log the error but don't fail - FTS might already exist
            logger.debug(f"FTS index creation skipped: {e}")

    async def _optimize(self) -> None:
        """Optimize the chunks table to refresh indexes."""
        # Skip optimization for LanceDB Cloud as it handles this automatically
        if Config.LANCEDB_URI and Config.LANCEDB_URI.startswith("db://"):
            return

        async with self._optimize_lock:
            try:
                self.store.chunks_table.optimize()
            except (RuntimeError, OSError) as e:
                # Handle "too many open files" and other resource errors gracefully
                logger.debug(
                    f"Table optimization skipped due to resource constraints: {e}"
                )

    async def create(self, entity: Chunk) -> Chunk:
        """Create a chunk in the database."""
        assert entity.document_id, "Chunk must have a document_id to be created"

        chunk_id = str(uuid4())

        # Generate embedding if not provided
        if entity.embedding is not None:
            embedding = entity.embedding
        else:
            embedding = await self.embedder.embed(entity.content)
        order_val = int(entity.order)

        chunk_record = self.store.ChunkRecord(
            id=chunk_id,
            document_id=entity.document_id,
            content=entity.content,
            metadata=json.dumps(
                {k: v for k, v in entity.metadata.items() if k != "order"}
            ),
            order=order_val,
            vector=embedding,
        )

        self.store.chunks_table.add([chunk_record])

        entity.id = chunk_id

        # Try to optimize if not currently locked (non-blocking)
        if not self._optimize_lock.locked():
            asyncio.create_task(self._optimize())

        return entity

    async def get_by_id(self, entity_id: str) -> Chunk | None:
        """Get a chunk by its ID."""
        results = list(
            self.store.chunks_table.search()
            .where(f"id = '{entity_id}'")
            .limit(1)
            .to_pydantic(self.store.ChunkRecord)
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
        """Update an existing chunk."""
        assert entity.id, "Chunk ID is required for update"

        embedding = await self.embedder.embed(entity.content)
        order_val = int(entity.order)

        self.store.chunks_table.update(
            where=f"id = '{entity.id}'",
            values={
                "document_id": entity.document_id,
                "content": entity.content,
                "metadata": json.dumps(
                    {k: v for k, v in entity.metadata.items() if k != "order"}
                ),
                "order": order_val,
                "vector": embedding,
            },
        )
        # Try to optimize if not currently locked (non-blocking)
        if not self._optimize_lock.locked():
            asyncio.create_task(self._optimize())

        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete a chunk by its ID."""
        chunk = await self.get_by_id(entity_id)
        if chunk is None:
            return False

        self.store.chunks_table.delete(f"id = '{entity_id}'")
        return True

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Chunk]:
        """List all chunks with optional pagination."""
        query = self.store.chunks_table.search()

        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        results = list(query.to_pydantic(self.store.ChunkRecord))

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

    async def create_chunks_for_document(
        self, document_id: str, document: DoclingDocument
    ) -> list[Chunk]:
        """Create chunks and embeddings for a document from DoclingDocument."""
        # Optionally preprocess markdown before chunking
        processed_document = document
        preprocessor_path = Config.MARKDOWN_PREPROCESSOR
        if preprocessor_path:
            try:
                pre_fn = load_callable(preprocessor_path)
                markdown = document.export_to_markdown()
                result = pre_fn(markdown)
                if inspect.isawaitable(result):
                    result = await result  # type: ignore[assignment]
                processed_markdown = result
                if not isinstance(processed_markdown, str):
                    raise ValueError("Preprocessor must return a markdown string")
                processed_document = text_to_docling_document(
                    processed_markdown, name="content.md"
                )
            except Exception as e:
                logger.error(
                    f"Failed to apply MARKDOWN_PREPROCESSOR '{preprocessor_path}': {e}. Proceeding without preprocessing."
                )
                raise e

        chunk_texts = await chunker.chunk(processed_document)

        embeddings = await self.embedder.embed(chunk_texts)

        # Prepare all chunk records for batch insertion
        chunk_records = []
        created_chunks = []

        for order, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk_id = str(uuid4())

            chunk_record = self.store.ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                metadata=json.dumps({}),
                order=order,
                vector=embedding,
            )
            chunk_records.append(chunk_record)

            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                metadata={},
                order=order,
            )
            created_chunks.append(chunk)

        # Batch insert all chunks at once
        if chunk_records:
            self.store.chunks_table.add(chunk_records)

        # Force optimization once at the end for bulk operations
        await self._optimize()
        return created_chunks

    async def delete_all(self) -> None:
        """Delete all chunks from the database."""
        # Drop and recreate table to clear all data
        self.store.db.drop_table("chunks")
        self.store.chunks_table = self.store.db.create_table(
            "chunks", schema=self.store.ChunkRecord
        )
        # Create FTS index on the new table with phrase query support
        self.store.chunks_table.create_fts_index(
            "content", replace=True, with_position=True, remove_stop_words=False
        )

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        chunks = await self.get_by_document_id(document_id)

        if not chunks:
            return False

        self.store.chunks_table.delete(f"document_id = '{document_id}'")
        return True

    async def search(
        self, query: str, limit: int = 5, search_type: str = "hybrid"
    ) -> list[tuple[Chunk, float]]:
        """Search for relevant chunks using the specified search method.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            search_type: Type of search - "vector", "fts", or "hybrid" (default).

        Returns:
            List of (chunk, score) tuples ordered by relevance.
        """
        if not query.strip():
            return []

        if search_type == "vector":
            query_embedding = await self.embedder.embed(query)

            results = self.store.chunks_table.search(
                query_embedding, query_type="vector", vector_column_name="vector"
            ).limit(limit)

            return await self._process_search_results(results)

        elif search_type == "fts":
            results = self.store.chunks_table.search(query, query_type="fts").limit(
                limit
            )
            return await self._process_search_results(results)

        else:  # hybrid (default)
            query_embedding = await self.embedder.embed(query)

            # Create RRF reranker
            reranker = RRFReranker()

            # Perform native hybrid search with RRF reranking
            results = (
                self.store.chunks_table.search(query_type="hybrid")
                .vector(query_embedding)
                .text(query)
                .rerank(reranker)
                .limit(limit)
            )
            return await self._process_search_results(results)

    async def get_by_document_id(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a specific document."""
        results = list(
            self.store.chunks_table.search()
            .where(f"document_id = '{document_id}'")
            .to_pydantic(self.store.ChunkRecord)
        )

        # Get document info
        doc_results = list(
            self.store.documents_table.search()
            .where(f"id = '{document_id}'")
            .limit(1)
            .to_pydantic(DocumentRecord)
        )

        doc_uri = doc_results[0].uri if doc_results else None
        doc_title = doc_results[0].title if doc_results else None
        doc_meta = doc_results[0].metadata if doc_results else "{}"

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

    async def get_adjacent_chunks(self, chunk: Chunk, num_adjacent: int) -> list[Chunk]:
        """Get adjacent chunks before and after the given chunk within the same document."""
        assert chunk.document_id, "Document id is required for adjacent chunk finding"

        chunk_order = chunk.order

        # Fetch chunks for the same document and filter by order proximity
        all_chunks = await self.get_by_document_id(chunk.document_id)

        adjacent_chunks: list[Chunk] = []
        for c in all_chunks:
            c_order = c.order
            if c.id != chunk.id and abs(c_order - chunk_order) <= num_adjacent:
                adjacent_chunks.append(c)

        return adjacent_chunks

    async def _process_search_results(self, query_result) -> list[tuple[Chunk, float]]:
        """Process search results into chunks with document info and scores."""
        chunks_with_scores = []

        # Get both arrow and pydantic results to access scores
        arrow_result = query_result.to_arrow()
        pydantic_results = list(query_result.to_pydantic(self.store.ChunkRecord))

        # Extract scores from arrow result based on search type
        scores = []
        column_names = arrow_result.column_names

        if "_distance" in column_names:
            # Vector search - distance (lower is better, convert to similarity)
            distances = arrow_result.column("_distance").to_pylist()
            scores = [max(0.0, 1.0 / (1.0 + dist)) for dist in distances]
        elif "_relevance_score" in column_names:
            # Hybrid search - relevance score (higher is better)
            scores = arrow_result.column("_relevance_score").to_pylist()
        elif "_score" in column_names:
            # FTS search - score (higher is better)
            scores = arrow_result.column("_score").to_pylist()
        else:
            raise ValueError("Unknown search result format, cannot extract scores")

        # Collect all unique document IDs for batch lookup
        document_ids = list(set(chunk.document_id for chunk in pydantic_results))

        # Batch fetch all documents at once
        documents_map = {}
        if document_ids:
            # Create a WHERE clause for all document IDs
            where_clause = " OR ".join(f"id = '{doc_id}'" for doc_id in document_ids)
            doc_results = list(
                self.store.documents_table.search()
                .where(where_clause)
                .to_pydantic(DocumentRecord)
            )
            documents_map = {doc.id: doc for doc in doc_results}

        for i, chunk_record in enumerate(pydantic_results):
            # Get document info from pre-fetched map
            doc = documents_map.get(chunk_record.document_id)
            doc_uri = doc.uri if doc else None
            doc_title = doc.title if doc else None
            doc_meta = doc.metadata if doc else "{}"

            md = json.loads(chunk_record.metadata)

            chunk = Chunk(
                id=chunk_record.id,
                document_id=chunk_record.document_id,
                content=chunk_record.content,
                metadata=md,
                order=chunk_record.order,
                document_uri=doc_uri,
                document_title=doc_title,
                document_meta=json.loads(doc_meta),
            )

            # Get score from arrow result
            score = scores[i] if i < len(scores) else 1.0

            chunks_with_scores.append((chunk, score))

        return chunks_with_scores
