from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG


class SearchResult(BaseModel):
    """Search result with both title and URI for A2A agent."""

    content: str = Field(description="The document text content")
    score: float = Field(description="Relevance score (higher is more relevant)")
    document_title: str | None = Field(
        description="Human-readable document title", default=None
    )
    document_uri: str = Field(description="Document URI/path for get_full_document")


class AgentDependencies(BaseModel):
    """Dependencies for the A2A conversational agent."""

    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG
