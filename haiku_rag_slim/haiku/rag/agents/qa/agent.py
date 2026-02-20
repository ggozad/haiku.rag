from dataclasses import dataclass

from pydantic_ai import Agent

from haiku.rag.agents.qa.prompts import QA_SYSTEM_PROMPT
from haiku.rag.agents.research.models import (
    Citation,
    RawSearchAnswer,
    resolve_citations,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.tools.search import create_search_toolset
from haiku.rag.utils import get_model


@dataclass
class _QARunDeps:
    client: HaikuRAG


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
        self._model_config = model_config
        self._system_prompt = system_prompt or QA_SYSTEM_PROMPT

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
        accumulated_results: list[SearchResult] = []
        search_toolset = create_search_toolset(
            self._config,
            base_filter=filter,
            tool_name="search_documents",
            on_results=accumulated_results.extend,
        )

        # Agent created per-call: toolset varies with filter, and Agent
        # construction is pure Python (no IO).
        agent: Agent[_QARunDeps, RawSearchAnswer] = Agent(  # ty: ignore[invalid-assignment]
            model=get_model(self._model_config, self._config),
            deps_type=_QARunDeps,
            output_type=RawSearchAnswer,
            output_retries=3,
            instructions=self._system_prompt,
            toolsets=[search_toolset],
            retries=3,
        )

        deps = _QARunDeps(client=self._client)
        result = await agent.run(question, deps=deps)
        output = result.output

        citations = resolve_citations(output.cited_chunks, accumulated_results)
        return output.answer, citations
