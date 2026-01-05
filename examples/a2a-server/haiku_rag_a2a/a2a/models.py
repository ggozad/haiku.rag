from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG


class A2AConfig(BaseModel):
    """Configuration for A2A (Agent-to-Agent) protocol server."""

    max_contexts: int = Field(
        default=1000, description="Maximum number of conversations to keep in memory"
    )


class AgentDependencies(BaseModel):
    """Dependencies for the A2A conversational agent."""

    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG
