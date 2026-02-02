from pydantic_ai import Agent
from pydantic_ai.output import ToolOutput

from haiku.rag.agents.qa.prompts import QA_SYSTEM_PROMPT
from haiku.rag.agents.research.models import (
    Citation,
    RawSearchAnswer,
    resolve_citations,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.search import SEARCH_NAMESPACE, SearchState, create_search_toolset
from haiku.rag.utils import get_model


class QuestionAnswerAgent:
    def __init__(
        self,
        client: HaikuRAG,
        model_config: ModelConfig,
        config: AppConfig | None = None,
        system_prompt: str | None = None,
    ):
        self._client = client
        self._config = config or Config
        self._agent: Agent[None, RawSearchAnswer] = Agent(
            model=get_model(model_config, self._config),
            output_type=ToolOutput(RawSearchAnswer, max_retries=3),
            instructions=system_prompt or QA_SYSTEM_PROMPT,
            retries=3,
        )

    async def answer(
        self, question: str, filter: str | None = None
    ) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG system.

        Args:
            question: The question to answer
            filter: SQL WHERE clause to filter documents

        Returns:
            Tuple of (answer text, list of resolved citations)
        """
        # Create context and search toolset for this run
        context = ToolContext()
        search_toolset = create_search_toolset(
            self._client,
            self._config,
            context=context,
            base_filter=filter,
            tool_name="search_documents",
        )

        result = await self._agent.run(question, toolsets=[search_toolset])
        output = result.output

        # Get search results from context for citation resolution
        search_state = context.get(SEARCH_NAMESPACE)
        search_results = (
            search_state.results if isinstance(search_state, SearchState) else []
        )

        citations = resolve_citations(output.cited_chunks, search_results)
        return output.answer, citations
