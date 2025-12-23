import json
from typing import TYPE_CHECKING, cast, overload
from uuid import uuid4

if TYPE_CHECKING:
    import pandas as pd
    from lancedb.query import LanceVectorQueryBuilder

from haiku.rag.store.engine import Store
from haiku.rag.store.models.raptor import RaptorNode


class RaptorNodeRepository:
    """Repository for RAPTOR node operations."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.embedder = store.embedder

    @overload
    async def create(self, entity: RaptorNode) -> RaptorNode: ...

    @overload
    async def create(self, entity: list[RaptorNode]) -> list[RaptorNode]: ...

    async def create(
        self, entity: RaptorNode | list[RaptorNode]
    ) -> RaptorNode | list[RaptorNode]:
        """Create one or more RAPTOR nodes in the database.

        Nodes must have embeddings set before calling this method.
        """
        self.store._assert_writable()

        if isinstance(entity, RaptorNode):
            assert entity.embedding is not None, "RaptorNode must have an embedding"

            node_id = str(uuid4())
            record = self.store.RaptorNodeRecord(
                id=node_id,
                content=entity.content,
                layer=entity.layer,
                cluster_id=entity.cluster_id,
                source_chunk_ids=json.dumps(entity.source_chunk_ids),
                vector=entity.embedding,
            )
            self.store.raptor_nodes_table.add([record])
            entity.id = node_id
            return entity

        nodes = entity
        if not nodes:
            return []

        for node in nodes:
            assert node.embedding is not None, "All RaptorNodes must have embeddings"

        records = []
        for node in nodes:
            node_id = str(uuid4())
            record = self.store.RaptorNodeRecord(
                id=node_id,
                content=node.content,
                layer=node.layer,
                cluster_id=node.cluster_id,
                source_chunk_ids=json.dumps(node.source_chunk_ids),
                vector=node.embedding,
            )
            records.append(record)
            node.id = node_id

        self.store.raptor_nodes_table.add(records)
        return nodes

    async def get_by_id(self, entity_id: str) -> RaptorNode | None:
        """Get a RAPTOR node by its ID."""
        results = list(
            self.store.raptor_nodes_table.search()
            .where(f"id = '{entity_id}'")
            .limit(1)
            .to_pydantic(self.store.RaptorNodeRecord)
        )

        if not results:
            return None

        record = results[0]
        return RaptorNode(
            id=record.id,
            content=record.content,
            layer=record.layer,
            cluster_id=record.cluster_id,
            source_chunk_ids=json.loads(record.source_chunk_ids),
            embedding=list(record.vector),
        )

    async def get_by_layer(self, layer: int) -> list[RaptorNode]:
        """Get all RAPTOR nodes at a specific layer."""
        results = list(
            self.store.raptor_nodes_table.search()
            .where(f"layer = {layer}")
            .to_pydantic(self.store.RaptorNodeRecord)
        )

        nodes: list[RaptorNode] = []
        for record in results:
            nodes.append(
                RaptorNode(
                    id=record.id,
                    content=record.content,
                    layer=record.layer,
                    cluster_id=record.cluster_id,
                    source_chunk_ids=json.loads(record.source_chunk_ids),
                    embedding=list(record.vector),
                )
            )
        return nodes

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[RaptorNode]:
        """List all RAPTOR nodes with optional pagination."""
        query = self.store.raptor_nodes_table.search()

        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        results = list(query.to_pydantic(self.store.RaptorNodeRecord))

        nodes: list[RaptorNode] = []
        for record in results:
            nodes.append(
                RaptorNode(
                    id=record.id,
                    content=record.content,
                    layer=record.layer,
                    cluster_id=record.cluster_id,
                    source_chunk_ids=json.loads(record.source_chunk_ids),
                    embedding=list(record.vector),
                )
            )
        return nodes

    async def delete_all(self) -> None:
        """Delete all RAPTOR nodes from the database."""
        self.store._assert_writable()
        self.store.db.drop_table("raptor_nodes")
        self.store.raptor_nodes_table = self.store.db.create_table(
            "raptor_nodes", schema=self.store.RaptorNodeRecord
        )

    async def count(self) -> int:
        """Count total number of RAPTOR nodes."""
        return self.store.raptor_nodes_table.count_rows()

    async def has_nodes(self) -> bool:
        """Check if any RAPTOR nodes exist."""
        return self.store.raptor_nodes_table.count_rows() > 0

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[RaptorNode, float]]:
        """Search for relevant RAPTOR nodes using vector similarity.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            List of (RaptorNode, score) tuples ordered by relevance.
        """
        if not query.strip():
            return []

        query_embedding = await self.embedder.embed(query)
        vector_query = cast(
            "LanceVectorQueryBuilder",
            self.store.raptor_nodes_table.search(
                query_embedding, query_type="vector", vector_column_name="vector"
            ),
        )
        results = vector_query.refine_factor(
            self.store._config.search.vector_refine_factor
        ).limit(limit)

        return await self._process_search_results(results)

    async def _process_search_results(
        self, query_result: "pd.DataFrame | LanceVectorQueryBuilder"
    ) -> list[tuple[RaptorNode, float]]:
        """Process search results into nodes with scores."""
        import pandas as pd

        if isinstance(query_result, pd.DataFrame):
            df = query_result
        else:
            df = query_result.to_pandas()

        if "_distance" in df.columns:
            scores = ((df["_distance"] + 1).rdiv(1)).clip(lower=0.0).tolist()
        else:
            scores = [1.0] * len(df)

        nodes_with_scores = []
        for i, (_, row) in enumerate(df.iterrows()):
            node = RaptorNode(
                id=str(row["id"]),
                content=str(row["content"]),
                layer=int(row["layer"]),
                cluster_id=int(row["cluster_id"]),
                source_chunk_ids=json.loads(str(row["source_chunk_ids"])),
                embedding=list(row["vector"]),
            )
            score = scores[i] if i < len(scores) else 1.0
            nodes_with_scores.append((node, score))

        return nodes_with_scores
