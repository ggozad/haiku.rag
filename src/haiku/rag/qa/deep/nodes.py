import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph import BaseNode, End, GraphRunContext

from haiku.rag.graph.common import get_model, log
from haiku.rag.graph.models import ResearchPlan, SearchAnswer
from haiku.rag.graph.prompts import PLAN_PROMPT, SEARCH_AGENT_PROMPT
from haiku.rag.qa.deep.dependencies import DeepQADependencies
from haiku.rag.qa.deep.models import DeepQAAnswer, DeepQAEvaluation
from haiku.rag.qa.deep.prompts import (
    DECISION_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_PROMPT_WITH_CITATIONS,
)
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState


@dataclass
class DeepQAPlanNode(BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[DeepQAState, DeepQADeps]
    ) -> BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]:
        state = ctx.state
        deps = ctx.deps

        log(deps, state, "\n[bold cyan]📋 Planning approach...[/bold cyan]")

        plan_agent = Agent(
            model=get_model(self.provider, self.model),
            output_type=ResearchPlan,
            instructions=(
                PLAN_PROMPT
                + "\n\nUse the gather_context tool once on the main question before planning."
            ),
            retries=3,
            deps_type=DeepQADependencies,
        )

        @plan_agent.tool
        async def gather_context(
            ctx2: RunContext[DeepQADependencies], query: str, limit: int = 6
        ) -> str:
            results = await ctx2.deps.client.search(query, limit=limit)
            expanded = await ctx2.deps.client.expand_context(results)
            return "\n\n".join(chunk.content for chunk, _ in expanded)

        prompt = (
            "Plan a focused approach for answering the main question.\n\n"
            f"Main question: {state.context.original_question}"
        )

        agent_deps = DeepQADependencies(
            client=deps.client,
            context=state.context,
            console=deps.console,
        )
        plan_result = await plan_agent.run(prompt, deps=agent_deps)
        state.context.sub_questions = list(plan_result.output.sub_questions)[
            : state.max_sub_questions
        ]

        log(deps, state, "\n[bold green]✅ Plan Created:[/bold green]")
        log(
            deps,
            state,
            f"   [bold]Main Question:[/bold] {state.context.original_question}",
        )
        log(deps, state, "   [bold]Sub-questions:[/bold]")
        for i, sq in enumerate(state.context.sub_questions, 1):
            log(deps, state, f"      {i}. {sq}")

        return DeepQASearchDispatchNode(self.provider, self.model)


@dataclass
class DeepQASearchDispatchNode(BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[DeepQAState, DeepQADeps]
    ) -> BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]:
        state = ctx.state
        deps = ctx.deps

        if not state.context.sub_questions:
            return DeepQADecisionNode(self.provider, self.model)

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
                deps_type=DeepQADependencies,
            )

            @agent.tool
            async def search_and_answer(
                ctx2: RunContext[DeepQADependencies], query: str, limit: int = 5
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

            agent_deps = DeepQADependencies(
                client=deps.client,
                context=state.context,
                console=deps.console,
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

        return DeepQASearchDispatchNode(self.provider, self.model)


@dataclass
class DeepQADecisionNode(BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[DeepQAState, DeepQADeps]
    ) -> BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]📊 Evaluating information sufficiency...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(self.provider, self.model),
            output_type=DeepQAEvaluation,
            instructions=DECISION_PROMPT,
            retries=3,
            deps_type=DeepQADependencies,
        )

        context_data = {
            "original_question": state.context.original_question,
            "gathered_answers": [
                {
                    "question": qa.query,
                    "answer": qa.answer,
                    "sources": qa.sources,
                }
                for qa in state.context.qa_responses
            ],
        }
        context_xml = format_as_xml(context_data, root_tag="gathered_information")

        prompt = (
            "Evaluate whether we have sufficient information to answer the question.\n\n"
            f"{context_xml}"
        )

        agent_deps = DeepQADependencies(
            client=deps.client,
            context=state.context,
            console=deps.console,
        )
        result = await agent.run(prompt, deps=agent_deps)
        evaluation = result.output

        state.iterations += 1

        log(deps, state, f"   [bold]Assessment:[/bold] {evaluation.reasoning}")
        status = "[green]Yes[/green]" if evaluation.is_sufficient else "[red]No[/red]"
        log(deps, state, f"   Sufficient: {status}")

        # Add new questions if not sufficient
        for new_q in evaluation.new_questions:
            if new_q not in state.context.sub_questions:
                state.context.sub_questions.append(new_q)

        if evaluation.new_questions:
            log(deps, state, "   [cyan]New questions:[/cyan]")
            for question in evaluation.new_questions:
                log(deps, state, f"   • {question}")

        # Decide next step
        if evaluation.is_sufficient or state.iterations >= state.max_iterations:
            if state.iterations >= state.max_iterations:
                log(
                    deps,
                    state,
                    f"\n[bold yellow]⚠️  Reached max iterations ({state.max_iterations})[/bold yellow]",
                )
            log(deps, state, "\n[bold green]✅ Moving to synthesis.[/bold green]")
            return DeepQASynthesizeNode(self.provider, self.model)

        log(
            deps,
            state,
            f"\n[bold cyan]🔄 Starting iteration {state.iterations + 1}...[/bold cyan]",
        )
        return DeepQASearchDispatchNode(self.provider, self.model)


@dataclass
class DeepQASynthesizeNode(BaseNode[DeepQAState, DeepQADeps, DeepQAAnswer]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[DeepQAState, DeepQADeps]
    ) -> End[DeepQAAnswer]:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]📝 Synthesizing final answer...[/bold cyan]",
        )

        prompt_template = (
            SYNTHESIS_PROMPT_WITH_CITATIONS
            if state.context.use_citations
            else SYNTHESIS_PROMPT
        )

        agent = Agent(
            model=get_model(self.provider, self.model),
            output_type=DeepQAAnswer,
            instructions=prompt_template,
            retries=3,
            deps_type=DeepQADependencies,
        )

        context_data = {
            "original_question": state.context.original_question,
            "sub_answers": [
                {
                    "question": qa.query,
                    "answer": qa.answer,
                    "sources": qa.sources,
                }
                for qa in state.context.qa_responses
            ],
        }
        context_xml = format_as_xml(context_data, root_tag="gathered_information")

        prompt = f"Synthesize a comprehensive answer to the original question.\n\n{context_xml}"

        agent_deps = DeepQADependencies(
            client=deps.client,
            context=state.context,
            console=deps.console,
        )
        result = await agent.run(prompt, deps=agent_deps)

        log(deps, state, "[bold green]✅ Answer complete![/bold green]")
        return End(result.output)
