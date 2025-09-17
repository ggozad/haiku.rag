from pydantic_ai import RunContext
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.run import AgentRunResult

from haiku.rag.research.base import BaseResearchAgent, SearchAnswer
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.research.prompts import SEARCH_AGENT_PROMPT


class SearchSpecialistAgent(BaseResearchAgent[SearchAnswer]):
    """Agent specialized in answering questions using RAG search."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, model, output_type=SearchAnswer)

    async def run(
        self, prompt: str, deps: ResearchDependencies, **kwargs
    ) -> AgentRunResult[SearchAnswer]:
        """Execute the agent and persist the QA pair in shared context.

        Pydantic AI enforces `SearchAnswer` as the output model; we just store
        the QA response with the last search results as sources.
        """
        result = await super().run(prompt, deps, **kwargs)

        if result.output:
            deps.context.add_qa_response(result.output)

        return result

    def get_system_prompt(self) -> str:
        return SEARCH_AGENT_PROMPT

    def register_tools(self) -> None:
        """Register search-specific tools."""

        @self.agent.tool
        async def search_and_answer(
            ctx: RunContext[ResearchDependencies],
            query: str,
            limit: int = 5,
        ) -> str:
            """Search the KB and return a concise context pack."""
            # Remove quotes from queries as this requires positional indexing in lancedb
            query = query.replace('"', "")
            search_results = await ctx.deps.client.search(query, limit=limit)
            expanded = await ctx.deps.client.expand_context(search_results)

            snippet_entries = [
                {
                    "text": chunk.content,
                    "score": score,
                    "document_uri": (chunk.document_uri or ""),
                }
                for chunk, score in expanded
            ]

            # Return an XML-formatted payload with the question and snippets.
            if snippet_entries:
                return format_as_xml(snippet_entries, root_tag="snippets")
            else:
                return (
                    f"No relevant information found in the knowledge base for: {query}"
                )
