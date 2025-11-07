from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG


class A2AConfig(BaseModel):
    """Configuration for A2A (Agent-to-Agent) protocol server."""

    max_contexts: int = Field(
        default=1000, description="Maximum number of conversations to keep in memory"
    )


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
