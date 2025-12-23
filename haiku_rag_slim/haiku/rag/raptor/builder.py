import logging
from typing import TYPE_CHECKING

import numpy as np

from haiku.rag.raptor.clustering import cluster_embeddings, group_into_clusters
from haiku.rag.raptor.summarizer import ClusterSummarizer
from haiku.rag.store.models.raptor import RaptorNode
from haiku.rag.store.repositories.raptor import RaptorNodeRepository

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

logger = logging.getLogger(__name__)


class RaptorTreeBuilder:
    """Builds the RAPTOR hierarchical summary tree."""

    def __init__(self, client: "HaikuRAG"):
        self._client = client
        self._config = client._config
        self._store = client.store
        self._embedder = self._store.embedder
        self._repo = RaptorNodeRepository(self._store)
        self._summarizer = ClusterSummarizer(self._config)

    async def build(self) -> int:
        """Build the RAPTOR tree from all chunks in the database.

        Recursively clusters chunks using UMAP + GMM, summarizes each cluster
        with an LLM, embeds the summaries, and stores them as RAPTOR nodes.
        Each layer's summaries become input for the next layer until max_depth
        is reached or too few items remain to cluster.

        Returns:
            Total number of RAPTOR nodes created
        """
        # Clear existing RAPTOR nodes
        await self._repo.delete_all()

        # Get all chunks with embeddings (search() without query returns all rows)
        chunks = list(
            self._store.chunks_table.search().to_pydantic(self._store.ChunkRecord)
        )

        if len(chunks) < self._config.raptor.min_cluster_size:
            logger.debug(
                f"Not enough chunks ({len(chunks)}) to build RAPTOR tree "
                f"(min_cluster_size={self._config.raptor.min_cluster_size})"
            )
            return 0

        # Layer 0 is chunks - start building from layer 1
        current_layer_texts = [c.content for c in chunks]
        current_layer_embeddings = np.array([c.vector for c in chunks])
        current_layer_chunk_ids = [[c.id] for c in chunks]  # Each chunk maps to itself

        total_nodes = 0
        layer = 1

        while layer <= self._config.raptor.max_depth:
            logger.debug(
                f"Building RAPTOR layer {layer} from {len(current_layer_texts)} items"
            )

            # Check if we have enough items to cluster
            if len(current_layer_texts) < self._config.raptor.min_cluster_size:
                logger.debug(f"Not enough items to cluster at layer {layer}")
                break

            # Cluster the current layer
            cluster_assignments = cluster_embeddings(
                current_layer_embeddings,
                reduction_dim=min(10, len(current_layer_embeddings) - 2),
                threshold=0.1,
                n_neighbors=self._config.raptor.umap_n_neighbors,
                min_dist=self._config.raptor.umap_min_dist,
            )

            # Group items by cluster
            text_clusters = group_into_clusters(
                current_layer_texts, cluster_assignments
            )
            chunk_id_clusters = group_into_clusters(
                current_layer_chunk_ids, cluster_assignments
            )

            # Filter out clusters that are too small
            valid_clusters = [
                (texts, chunk_ids)
                for texts, chunk_ids in zip(text_clusters, chunk_id_clusters)
                if len(texts) >= self._config.raptor.min_cluster_size
            ]

            if not valid_clusters:
                logger.debug(f"No valid clusters at layer {layer}")
                break

            # Create summary nodes for each cluster
            next_layer_texts = []
            next_layer_embeddings = []
            next_layer_chunk_ids = []

            for cluster_idx, (texts, chunk_ids_list) in enumerate(valid_clusters):
                # Summarize the cluster
                summary = await self._summarizer.summarize(texts)

                # Embed the summary
                embedding = await self._embedder.embed(summary)

                # Flatten source chunk IDs
                source_chunk_ids = []
                for ids in chunk_ids_list:
                    source_chunk_ids.extend(ids)
                source_chunk_ids = list(set(source_chunk_ids))  # Dedupe

                # Create the node
                node = RaptorNode(
                    content=summary,
                    layer=layer,
                    cluster_id=cluster_idx,
                    source_chunk_ids=source_chunk_ids,
                    embedding=embedding,
                )
                await self._repo.create(node)
                total_nodes += 1

                # Add to next layer inputs
                next_layer_texts.append(summary)
                next_layer_embeddings.append(embedding)
                next_layer_chunk_ids.append(source_chunk_ids)

            logger.debug(f"Created {len(valid_clusters)} nodes at layer {layer}")

            # Prepare for next layer
            current_layer_texts = next_layer_texts
            current_layer_embeddings = np.array(next_layer_embeddings)
            current_layer_chunk_ids = next_layer_chunk_ids
            layer += 1

        logger.debug(f"RAPTOR tree built with {total_nodes} total nodes")
        return total_nodes
