import json
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument
from lancedb.rerankers import RRFReranker

from haiku.rag.chunker import chunker
from haiku.rag.embeddings import get_embedder
from haiku.rag.store.engine import DocumentRecord, Store
from haiku.rag.store.models.chunk import Chunk


class ChunkRepository:
    """Repository for Chunk operations."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.embedder = get_embedder()

    def _ensure_fts_index(self) -> None:
        """Ensure FTS index exists on the content column."""
        try:
            self.store.chunks_table.create_fts_index("content", replace=True)
        except Exception:
            pass

    async def create(self, entity: Chunk) -> Chunk:
        """Create a chunk in the database."""
        assert entity.document_id, "Chunk must have a document_id to be created"

        chunk_id = str(uuid4())

        # Generate embedding if not provided
        if entity.embedding is not None:
            embedding = entity.embedding
        else:
            embedding = await self.embedder.embed(entity.content)

        chunk_record = self.store.ChunkRecord(
            id=chunk_id,
            document_id=entity.document_id,
            content=entity.content,
            metadata=json.dumps(entity.metadata),
            vector=embedding,
        )

        self.store.chunks_table.add([chunk_record])

        entity.id = chunk_id
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
        return Chunk(
            id=chunk_record.id,
            document_id=chunk_record.document_id,
            content=chunk_record.content,
            metadata=json.loads(chunk_record.metadata) if chunk_record.metadata else {},
        )

    async def update(self, entity: Chunk) -> Chunk:
        """Update an existing chunk."""
        assert entity.id, "Chunk ID is required for update"

        embedding = await self.embedder.embed(entity.content)

        self.store.chunks_table.update(
            where=f"id = '{entity.id}'",
            values={
                "document_id": entity.document_id,
                "content": entity.content,
                "metadata": json.dumps(entity.metadata),
                "vector": embedding,
            },
        )

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

        return [
            Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=json.loads(chunk.metadata) if chunk.metadata else {},
            )
            for chunk in results
        ]

    async def create_chunks_for_document(
        self, document_id: str, document: DoclingDocument
    ) -> list[Chunk]:
        """Create chunks and embeddings for a document from DoclingDocument."""
        chunk_texts = await chunker.chunk(document)
        created_chunks = []

        for order, chunk_text in enumerate(chunk_texts):
            chunk = Chunk(
                document_id=document_id, content=chunk_text, metadata={"order": order}
            )
            created_chunk = await self.create(chunk)
            created_chunks.append(created_chunk)

        return created_chunks

    async def delete_all(self) -> bool:
        """Delete all chunks from the database."""
        try:
            count = len(
                list(
                    self.store.chunks_table.search()
                    .limit(1)
                    .to_pydantic(self.store.ChunkRecord)
                )
            )
            if count > 0:
                # Drop and recreate table to clear all data
                self.store.db.drop_table("chunks")
                self.store.chunks_table = self.store.db.create_table(
                    "chunks", schema=self.store.ChunkRecord
                )
                return True
            return False
        except Exception:
            return False

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

            results = (
                self.store.chunks_table.search(query_embedding, query_type="vector")
                .limit(limit)
                .to_pydantic(self.store.ChunkRecord)
            )

            return await self._process_search_results(results)

        elif search_type == "fts":
            # Ensure FTS index exists
            self._ensure_fts_index()

            results = (
                self.store.chunks_table.search(query, query_type="fts")
                .limit(limit)
                .to_pydantic(self.store.ChunkRecord)
            )
            return await self._process_search_results(results)

        else:  # hybrid (default)
            # Ensure FTS index exists for hybrid search
            self._ensure_fts_index()

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
                .to_pydantic(self.store.ChunkRecord)
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
        doc_meta = doc_results[0].metadata if doc_results else "{}"

        # Sort by order in metadata
        chunks = [
            Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=json.loads(chunk.metadata) if chunk.metadata else {},
                document_uri=doc_uri,
                document_meta=json.loads(doc_meta) if doc_meta else {},
            )
            for chunk in results
        ]

        chunks.sort(key=lambda c: c.metadata.get("order", 0))
        return chunks

    async def get_adjacent_chunks(self, chunk: Chunk, num_adjacent: int) -> list[Chunk]:
        """Get adjacent chunks before and after the given chunk within the same document."""
        assert chunk.document_id, "Document id is required for adjacent chunk finding"

        chunk_order = chunk.metadata.get("order")
        if chunk_order is None:
            return []

        # Get all chunks for the document
        all_chunks = await self.get_by_document_id(chunk.document_id)

        # Filter to adjacent chunks
        adjacent_chunks = []
        for c in all_chunks:
            c_order = c.metadata.get("order", 0)
            if c.id != chunk.id and abs(c_order - chunk_order) <= num_adjacent:
                adjacent_chunks.append(c)

        return adjacent_chunks

    async def _process_search_results(self, results) -> list[tuple[Chunk, float]]:
        """Process search results into chunks with document info and scores."""
        chunks_with_scores = []

        for chunk_record in results:
            # Get document info
            doc_results = list(
                self.store.documents_table.search()
                .where(f"id = '{chunk_record.document_id}'")
                .limit(1)
                .to_pydantic(DocumentRecord)
            )

            doc_uri = doc_results[0].uri if doc_results else None
            doc_meta = doc_results[0].metadata if doc_results else "{}"

            chunk = Chunk(
                id=chunk_record.id,
                document_id=chunk_record.document_id,
                content=chunk_record.content,
                metadata=json.loads(chunk_record.metadata)
                if chunk_record.metadata
                else {},
                document_uri=doc_uri,
                document_meta=json.loads(doc_meta) if doc_meta else {},
            )

            # Get distance score - LanceDB returns _distance (lower is better)
            distance = getattr(chunk_record, "_distance", 1.0)

            # Convert distance to similarity score (higher is better)
            # Using exponential decay to convert distance to similarity
            score = max(0.0, 1.0 / (1.0 + distance))

            chunks_with_scores.append((chunk, score))

        return chunks_with_scores
