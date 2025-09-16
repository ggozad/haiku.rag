from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.run import AgentRunResult

from haiku.rag.config import Config
from haiku.rag.research.dependencies import ResearchDependencies

T = TypeVar("T")


class BaseResearchAgent(ABC, Generic[T]):
    """Base class for all research agents."""

    def __init__(
        self,
        provider: str,
        model: str,
        output_type: type[T],
    ):
        self.provider = provider
        self.model = model
        self.output_type = output_type

        model_obj = self._get_model(provider, model)

        self._agent = Agent(
            model=model_obj,
            deps_type=ResearchDependencies,
            output_type=self.output_type,
            system_prompt=self.get_system_prompt(),
        )

        # Register tools
        self.register_tools()

    def _get_model(self, provider: str, model: str):
        """Get the appropriate model object for the provider."""
        if provider == "ollama":
            return OpenAIChatModel(
                model_name=model,
                provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
            )
        elif provider == "vllm":
            return OpenAIChatModel(
                model_name=model,
                provider=OpenAIProvider(
                    base_url=f"{Config.VLLM_RESEARCH_BASE_URL or Config.VLLM_QA_BASE_URL}/v1",
                    api_key="none",
                ),
            )
        else:
            # For all other providers, use the provider:model format
            return f"{provider}:{model}"

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def register_tools(self) -> None:
        """Register agent-specific tools."""
        pass

    async def run(
        self, prompt: str, deps: ResearchDependencies, **kwargs
    ) -> AgentRunResult[T]:
        """Execute the agent."""
        return await self._agent.run(prompt, deps=deps, **kwargs)

    @property
    def agent(self) -> Agent[ResearchDependencies, T]:
        """Access the underlying Pydantic AI agent."""
        return self._agent


class SearchResult(BaseModel):
    """Standard search result format."""

    content: str
    score: float
    document_uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchOutput(BaseModel):
    """Standard research output format."""

    summary: str
    detailed_findings: list[str]
    sources: list[str]
    confidence: float
