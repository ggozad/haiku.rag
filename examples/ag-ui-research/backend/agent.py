"""Research assistant agent with graph integration."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import HumanDecision, ResearchDeps, ResearchState
from haiku.rag.utils import get_model

if TYPE_CHECKING:
    from haiku.rag.graph.agui.emitter import AGUIEmitter
    from haiku.rag.graph.research.models import ResearchReport

# Load config
config_path = Path("/app/haiku.rag.yaml")
Config = (
    AppConfig.model_validate(load_yaml_config(config_path))
    if config_path.exists()
    else AppConfig()
)


@dataclass
class ActiveResearch:
    """Tracks state for active research awaiting human decision."""

    queue: asyncio.Queue[HumanDecision]
    sub_questions: list[str] = field(default_factory=list)
    qa_responses: list[dict] = field(default_factory=list)
    original_question: str = ""


# Global registry of active research by thread_id
_active_research: dict[str, ActiveResearch] = {}


@dataclass
class AgentDeps:
    """Dependencies for research agent."""

    client: HaikuRAG
    agui_emitter: "AGUIEmitter[ResearchState, ResearchReport] | None" = None
    search_filter: str | None = None
    thread_id: str | None = None


model = get_model(Config.research.model, Config)

agent = Agent(
    model,
    deps_type=AgentDeps,
    system_prompt="""You are an advanced research assistant powered by haiku.rag.

CRITICAL RULES:
1. For greetings (hi, hello, hey, etc) or casual chat: respond directly WITHOUT using any tools
2. For questions about yourself or the system: respond directly WITHOUT using any tools
3. For substantive questions requiring information: ALWAYS use the run_research tool
4. NEVER answer substantive questions from your own knowledge - always use the tool

How to decide:
- "Hi" / "Hello" / "How are you?" -> Respond directly, NO tools
- "What can you do?" -> Respond directly, NO tools
- "How does X work in the codebase?" -> Use run_research tool
- "Tell me about Y" -> Use run_research tool

When you use run_research, the graph will decompose questions, search the knowledge base,
and generate a comprehensive report.

Be friendly and conversational in all responses.""",
)


@agent.tool
async def run_research(ctx: RunContext[AgentDeps], question: str) -> str:
    """Execute research graph on a substantive question.

    Use for questions requiring knowledge base search.
    DO NOT use for greetings or casual conversation.
    """
    if ctx.deps.agui_emitter:
        ctx.deps.agui_emitter.log(f"Starting research on: {question}")

    # Create queue for human decisions
    queue: asyncio.Queue[HumanDecision] = asyncio.Queue()

    # Build interactive graph
    graph = build_research_graph(Config, interactive=True)
    context = ResearchContext(original_question=question)
    state = ResearchState.from_config(context=context, config=Config)
    state.search_filter = ctx.deps.search_filter

    # Register active research for decision endpoint to find
    thread_id = ctx.deps.thread_id
    if thread_id:
        _active_research[thread_id] = ActiveResearch(
            queue=queue,
            sub_questions=[],
            qa_responses=[],
            original_question=question,
        )

    graph_deps = ResearchDeps(
        client=ctx.deps.client,
        agui_emitter=ctx.deps.agui_emitter,
        human_input_queue=queue,
        interactive=True,
    )

    try:
        result = await graph.run(state=state, deps=graph_deps)

        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.log("Research complete!")

        return f"""Research completed successfully!

Question: {question}

Executive Summary: {result.executive_summary}

Main Findings:
{chr(10).join(f"- {finding}" for finding in result.main_findings[:3])}

Conclusions:
{chr(10).join(f"- {conclusion}" for conclusion in result.conclusions[:2])}

Confidence: {f"{state.last_eval.confidence_score:.0%}" if state.last_eval else "N/A"}
Iterations completed: {state.iterations}

The full research report with all citations has been provided to the user.
"""

    except Exception as e:
        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.log(f"Research error: {str(e)}")
        return f"I encountered an error while researching: {str(e)}"
    finally:
        # Cleanup
        if thread_id and thread_id in _active_research:
            del _active_research[thread_id]
