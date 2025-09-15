"""Search specialist agent that answers questions using RAG."""

from typing import Any

from pydantic_ai import RunContext

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import ResearchDependencies


class SearchSpecialistAgent(BaseResearchAgent):
    """Agent specialized in answering questions using RAG search."""

    def __init__(self, provider: str, model: str):
        # Output is a string answer, like the QA agent
        super().__init__(provider, model, output_type=str)

    async def run(self, prompt: str, deps: ResearchDependencies, **kwargs) -> Any:
        """Execute the agent and store QA response in context."""
        # Run the base agent
        result = await super().run(prompt, deps, **kwargs)

        # Store the QA response if we got an answer
        if hasattr(result, "output") and result.output:
            # Get the sources from the last search (which the tool just stored)
            if deps.context.search_results:
                last_search = deps.context.search_results[-1]
                sources = last_search.get("results", [])
                deps.context.add_qa_response(prompt, result.output, sources)

        return result

    def get_system_prompt(self) -> str:
        return """You are a search and question-answering specialist.

        Your role is to:
        1. Search the knowledge base for relevant information
        2. Analyze the retrieved documents
        3. Provide a comprehensive answer to the question
        4. Base your answer strictly on the information found

        Use the search_and_answer tool to retrieve relevant documents and formulate your response.
        Be thorough and specific in your answers, citing relevant information from the sources."""

    def register_tools(self) -> None:
        """Register search-specific tools."""

        @self.agent.tool
        async def search_and_answer(
            ctx: RunContext[ResearchDependencies],
            query: str,
            limit: int = 5,
        ) -> str:
            """Search for information and provide context for answering the question."""
            # Use the default hybrid search
            search_results = await ctx.deps.client.search(query, limit=limit)

            # Expand context for better relevance
            expanded = await ctx.deps.client.expand_context(search_results)

            # Convert to SearchResult for context storage
            from haiku.rag.research.base import SearchResult

            results_for_context = []
            context_texts = []

            for chunk, score in expanded:
                results_for_context.append(
                    SearchResult(
                        content=chunk.content,
                        score=score,
                        document_uri=chunk.document_uri or "",
                        metadata={"chunk_id": chunk.id} if chunk.id else {},
                    )
                )
                context_texts.append(chunk.content)

            # Store raw search results for analysis by other agents
            ctx.deps.context.add_search_result(query, results_for_context)

            # Format context for the LLM to answer the question
            if context_texts:
                context = "\n\n---\n\n".join(context_texts)
                return f"Based on the following information from the knowledge base:\n\n{context}\n\nAnswer the question: {query}"
            else:
                return (
                    f"No relevant information found in the knowledge base for: {query}"
                )
