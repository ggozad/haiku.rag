from pydantic_ai import Agent
from pydantic_ai.format_prompt import format_as_xml
from pydantic_graph.beta import Graph, GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph.common import get_model
from haiku.rag.graph.common.models import SearchAnswer
from haiku.rag.graph.common.nodes import create_plan_node, create_search_node
from haiku.rag.graph.deep_qa.dependencies import DeepQADependencies
from haiku.rag.graph.deep_qa.models import DeepQAAnswer, DeepQAEvaluation
from haiku.rag.graph.deep_qa.prompts import (
    DECISION_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_PROMPT_WITH_CITATIONS,
)
from haiku.rag.graph.deep_qa.state import DeepQADeps, DeepQAState


def build_deep_qa_graph(
    config: AppConfig = Config,
) -> Graph[DeepQAState, DeepQADeps, None, DeepQAAnswer]:
    """Build the Deep QA graph.

    Args:
        config: AppConfig object (uses config.qa for provider, model, and graph parameters)

    Returns:
        Configured Deep QA graph
    """
    provider = config.qa.provider
    model = config.qa.model
    g = GraphBuilder(
        state_type=DeepQAState,
        deps_type=DeepQADeps,
        output_type=DeepQAAnswer,
    )

    # Create and register the plan node using the factory
    plan = g.step(
        create_plan_node(
            provider=provider,
            model=model,
            deps_type=DeepQADependencies,  # type: ignore[arg-type]
            activity_message="Planning approach",
            output_retries=None,  # Deep QA doesn't use output_retries
        )
    )  # type: ignore[arg-type]

    # Create and register the search_one node using the factory
    search_one = g.step(
        create_search_node(
            provider=provider,
            model=model,
            deps_type=DeepQADependencies,  # type: ignore[arg-type]
            with_step_wrapper=False,  # Deep QA doesn't wrap with agui_emitter step
            success_message_format="Answered: {sub_q}",
            handle_exceptions=True,
        )
    )  # type: ignore[arg-type]

    @g.step
    async def get_batch(
        ctx: StepContext[DeepQAState, DeepQADeps, None | bool],
    ) -> list[str] | None:
        """Get all remaining questions for this iteration."""
        state = ctx.state

        if not state.context.sub_questions:
            return None

        # Take ALL remaining questions - max_concurrency controls parallel execution within .map()
        batch = list(state.context.sub_questions)
        state.context.sub_questions.clear()
        return batch

    @g.step
    async def decide(
        ctx: StepContext[DeepQAState, DeepQADeps, list[SearchAnswer]],
    ) -> bool:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("decide")
            deps.agui_emitter.update_activity(
                "evaluating", "Evaluating information sufficiency"
            )

        try:
            agent = Agent(
                model=get_model(provider, model),
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
            )
            result = await agent.run(prompt, deps=agent_deps)
            evaluation = result.output

            state.iterations += 1

            for new_q in evaluation.new_questions:
                if new_q not in state.context.sub_questions:
                    state.context.sub_questions.append(new_q)

            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                status = "sufficient" if evaluation.is_sufficient else "insufficient"
                deps.agui_emitter.update_activity(
                    "evaluating",
                    f"Information {status} after {state.iterations} iteration(s)",
                )

            should_continue = (
                not evaluation.is_sufficient and state.iterations < state.max_iterations
            )

            return should_continue
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def synthesize(
        ctx: StepContext[DeepQAState, DeepQADeps, None | bool],
    ) -> DeepQAAnswer:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("synthesize")
            deps.agui_emitter.update_activity(
                "synthesizing", "Synthesizing final answer"
            )

        try:
            prompt_template = (
                SYNTHESIS_PROMPT_WITH_CITATIONS
                if state.context.use_citations
                else SYNTHESIS_PROMPT
            )

            agent = Agent(
                model=get_model(provider, model),
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
            )
            result = await agent.run(prompt, deps=agent_deps)

            if deps.agui_emitter:
                deps.agui_emitter.update_activity("synthesizing", "Answer complete")

            return result.output
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    # Build the graph structure
    collect_answers = g.join(
        reduce_list_append,
        initial_factory=list[SearchAnswer],
    )

    g.add(
        g.edge_from(g.start_node).to(plan),
        g.edge_from(plan).to(get_batch),
    )

    # Branch based on whether we have questions
    g.add(
        g.edge_from(get_batch).to(
            g.decision()
            .branch(g.match(list).label("Has questions").map().to(search_one))
            .branch(g.match(type(None)).label("No questions").to(synthesize))
        ),
        g.edge_from(search_one).to(collect_answers),
        g.edge_from(collect_answers).to(decide),
    )

    # Branch based on decision
    g.add(
        g.edge_from(decide).to(
            g.decision()
            .branch(
                g.match(bool, matches=lambda x: x).label("Continue QA").to(get_batch)
            )
            .branch(
                g.match(bool, matches=lambda x: not x)
                .label("Done with QA")
                .to(synthesize)
            )
        ),
        g.edge_from(synthesize).to(g.end_node),
    )

    return g.build()
