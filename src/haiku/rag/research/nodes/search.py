import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph import BaseNode, GraphRunContext

from haiku.rag.research.common import get_model, log
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.research.models import ResearchReport, SearchAnswer
from haiku.rag.research.prompts import SEARCH_AGENT_PROMPT
from haiku.rag.research.state import ResearchDeps, ResearchState


@dataclass
class SearchDispatchNode(BaseNode[ResearchState, ResearchDeps, ResearchReport]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> BaseNode[ResearchState, ResearchDeps, ResearchReport]:
        state = ctx.state
        deps = ctx.deps
        if not state.context.sub_questions:
            from haiku.rag.research.nodes.analysis import AnalyzeInsightsNode

            return AnalyzeInsightsNode(self.provider, self.model)

        # Take up to max_concurrency questions and answer them concurrently
        take = max(1, state.max_concurrency)
        batch: list[str] = []
        while state.context.sub_questions and len(batch) < take:
            batch.append(state.context.sub_questions.pop(0))

        async def answer_one(sub_q: str) -> SearchAnswer | None:
            log(
                deps,
                state,
                f"\n[bold cyan]🔍 Searching & Answering:[/bold cyan] {sub_q}",
            )
            agent = Agent(
                model=get_model(self.provider, self.model),
                output_type=ToolOutput(SearchAnswer, max_retries=3),
                instructions=SEARCH_AGENT_PROMPT,
                retries=3,
                deps_type=ResearchDependencies,
            )

            @agent.tool
            async def search_and_answer(
                ctx2: RunContext[ResearchDependencies], query: str, limit: int = 5
            ) -> str:
                search_results = await ctx2.deps.client.search(query, limit=limit)
                expanded = await ctx2.deps.client.expand_context(search_results)

                entries: list[dict[str, Any]] = [
                    {
                        "text": chunk.content,
                        "score": score,
                        "document_uri": (
                            chunk.document_title or chunk.document_uri or ""
                        ),
                    }
                    for chunk, score in expanded
                ]
                if not entries:
                    return f"No relevant information found in the knowledge base for: {query}"

                return format_as_xml(entries, root_tag="snippets")

            agent_deps = ResearchDependencies(
                client=deps.client,
                context=state.context,
                console=deps.console,
                stream=deps.stream,
            )
            try:
                result = await agent.run(sub_q, deps=agent_deps)
            except Exception as e:
                log(deps, state, f"[red]Search failed:[/red] {e}")
                return None

            return result.output

        answers = await asyncio.gather(*(answer_one(q) for q in batch))
        for ans in answers:
            if ans is None:
                continue
            state.context.add_qa_response(ans)
            preview = ans.answer[:150] + ("…" if len(ans.answer) > 150 else "")
            log(deps, state, f"   [green]✓[/green] {preview}")

        return SearchDispatchNode(self.provider, self.model)
