from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT, QA_SYSTEM_PROMPT_WITH_CITATIONS


class SearchResult(BaseModel):
    content: str = Field(description="The document text content")
    score: float = Field(description="Relevance score (higher is more relevant)")
    document_uri: str = Field(
        description="Source title (if available) or URI/path of the document"
    )


class Dependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG


class QuestionAnswerAgent:
    def __init__(
        self,
        client: HaikuRAG,
        provider: str,
        model: str,
        use_citations: bool = False,
        q: float = 0.0,
        system_prompt: str | None = None,
    ):
        self._client = client

        if system_prompt is None:
            system_prompt = (
                QA_SYSTEM_PROMPT_WITH_CITATIONS if use_citations else QA_SYSTEM_PROMPT
            )
        model_obj = self._get_model(provider, model)

        self._agent = Agent(
            model=model_obj,
            deps_type=Dependencies,
            system_prompt=system_prompt,
            retries=3,
        )

        @self._agent.tool
        async def search_documents(
            ctx: RunContext[Dependencies],
            query: str,
            limit: int = 3,
        ) -> list[SearchResult]:
            """Search the knowledge base for relevant documents."""
            search_results = await ctx.deps.client.search(query, limit=limit)
            expanded_results = await ctx.deps.client.expand_context(search_results)

            return [
                SearchResult(
                    content=chunk.content,
                    score=score,
                    document_uri=(chunk.document_title or chunk.document_uri or ""),
                )
                for chunk, score in expanded_results
            ]

    def _get_model(self, provider: str, model: str):
        """Get the appropriate model object for the provider."""
        if provider == "ollama":
            return OpenAIChatModel(
                model_name=model,
                provider=OllamaProvider(
                    base_url=f"{Config.providers.ollama.base_url}/v1"
                ),
            )
        elif provider == "vllm":
            return OpenAIChatModel(
                model_name=model,
                provider=OpenAIProvider(
                    base_url=f"{Config.providers.vllm.qa_base_url}/v1", api_key="none"
                ),
            )
        else:
            # For all other providers, use the provider:model format
            return f"{provider}:{model}"

    async def answer(self, question: str) -> str:
        """Answer a question using the RAG system."""
        deps = Dependencies(client=self._client)
        result = await self._agent.run(question, deps=deps)
        return result.output
