from pydantic import BaseModel


class RaptorNode(BaseModel):
    """A summary node in the RAPTOR hierarchical tree.

    Original chunks (conceptually layer 0) are stored in the chunks table.
    This table only contains summary nodes (layer >= 1).

    Attributes:
        id: Unique identifier for the node
        content: Summary text content
        layer: Tree layer (1+ for summaries)
        cluster_id: Cluster number within the layer
        source_chunk_ids: Original chunk IDs this node ultimately derives from
        embedding: Vector embedding of the content
    """

    id: str | None = None
    content: str
    layer: int
    cluster_id: int
    source_chunk_ids: list[str] = []
    embedding: list[float] | None = None
