from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config

QUERY_EXPANSION_PROMPT = """You are a search query expansion assistant. Given a user's search query, generate alternative phrasings and perspectives that capture the same information need.

Guidelines:
- Generate semantically equivalent queries with different word choices
- Consider synonyms, related terms, and different perspectives
- Keep queries concise and focused
- Do NOT include the original query in your output
- Generate exactly the requested number of alternative queries"""


class QueryVariants(BaseModel):
    queries: list[str] = Field(
        description="List of alternative query phrasings to improve search recall"
    )


class QueryExpander:
    """Expands queries into multiple variants using an LLM."""

    def __init__(self, provider: str, model: str, num_queries: int = 3):
        """Initialize the query expander.

        Args:
            provider: LLM provider (e.g., "ollama", "openai", "vllm")
            model: Model name to use for expansion
            num_queries: Number of query variants to generate (default: 3)
        """
        self.provider = provider
        self.model = model
        self.num_queries = num_queries

        model_obj = self._get_model(provider, model)

        self._agent = Agent(
            model=model_obj,
            output_type=QueryVariants,
            system_prompt=QUERY_EXPANSION_PROMPT,
            retries=2,
        )

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
                    base_url=f"{Config.VLLM_QA_BASE_URL}/v1", api_key="none"
                ),
            )
        else:
            # For all other providers, use the provider:model format
            return f"{provider}:{model}"

    async def expand(self, query: str) -> list[str]:
        """Generate alternative query variants.

        Args:
            query: The original search query

        Returns:
            List of alternative query strings
        """
        prompt = f"Generate {self.num_queries} alternative phrasings for this search query:\n\n{query}"

        result = await self._agent.run(prompt)
        return result.output.queries[: self.num_queries]
