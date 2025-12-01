from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.config.models import ModelConfig
from haiku.rag.graph.common import get_model
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT, QA_SYSTEM_PROMPT_WITH_CITATIONS


class ToolSearchResult(BaseModel):
    """Search result model exposed to the LLM tool."""

    content: str = Field(description="The document text content")
    score: float = Field(description="Relevance score (higher is more relevant)")
    document_uri: str = Field(description="The URI/path of the source document")
    document_title: str | None = Field(
        default=None, description="The title of the document (if available)"
    )
    page_numbers: list[int] = Field(
        default=[], description="Page numbers where this content appears"
    )
    headings: list[str] | None = Field(
        default=None, description="Section heading hierarchy for this content"
    )


class Dependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG


class QuestionAnswerAgent:
    def __init__(
        self,
        client: HaikuRAG,
        model_config: ModelConfig,
        use_citations: bool = False,
        q: float = 0.0,
        system_prompt: str | None = None,
    ):
        self._client = client

        if system_prompt is None:
            system_prompt = (
                QA_SYSTEM_PROMPT_WITH_CITATIONS if use_citations else QA_SYSTEM_PROMPT
            )
        model_obj = get_model(model_config, Config)

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
            limit: int = 5,
        ) -> list[ToolSearchResult]:
            """Search the knowledge base for relevant documents."""
            results = await ctx.deps.client.search(query, limit=limit)
            results = await ctx.deps.client.expand_context(results)
            return [
                ToolSearchResult(
                    content=r.content,
                    score=r.score,
                    document_uri=(r.document_uri or ""),
                    document_title=r.document_title,
                    page_numbers=r.page_numbers,
                    headings=r.headings,
                )
                for r in results
            ]

    async def answer(self, question: str) -> str:
        """Answer a question using the RAG system."""
        deps = Dependencies(client=self._client)
        result = await self._agent.run(question, deps=deps)
        return result.output
