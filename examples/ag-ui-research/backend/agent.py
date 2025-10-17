"""Pydantic AI research agent for haiku.rag with AG-UI protocol."""

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps

from haiku.rag.config import Config
from haiku.rag.graph.common import get_model


class ResearchState(BaseModel):
    """Shared state between research agent and frontend."""

    question: str = ""
    status: str = "idle"
    current_iteration: int = 0
    max_iterations: int = 2
    confidence: float = 0.0
    plan: list[dict] = []
    findings: list[dict] = []
    final_report: dict | None = None


def _as_state_snapshot(ctx: RunContext[StateDeps[ResearchState]]) -> StateSnapshotEvent:
    """Helper to create a state snapshot event for AG-UI."""
    return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


def create_agent(
    qa_provider: str = Config.QA_PROVIDER, qa_model: str = Config.QA_MODEL
) -> Agent[StateDeps[ResearchState], str]:
    """Create and configure the research agent.

    Args:
        qa_provider: QA provider for the agent (default: from Config.QA_PROVIDER)
        qa_model: Model name to use (default: from Config.QA_MODEL)
    """
    agent = Agent(
        model=get_model(qa_provider, qa_model),
        deps_type=StateDeps[ResearchState],
        instructions="""You are a research assistant powered by haiku.rag.

You help users conduct deep research on complex questions by:
- Breaking down questions into sub-questions
- Searching through a knowledge base
- Evaluating findings for completeness and confidence
- Synthesizing comprehensive reports with citations

The state is shared with the frontend application, showing research progress in real-time.

Currently, tools are placeholder stubs. Full integration with haiku.rag research pipeline
will be implemented in the next phase.""",
    )

    @agent.tool
    async def get_research_status(ctx: RunContext[StateDeps[ResearchState]]) -> dict:
        """Get the current research state and progress."""
        return {
            "question": ctx.deps.state.question,
            "status": ctx.deps.state.status,
            "iteration": ctx.deps.state.current_iteration,
            "max_iterations": ctx.deps.state.max_iterations,
            "confidence": ctx.deps.state.confidence,
            "has_plan": len(ctx.deps.state.plan) > 0,
            "findings_count": len(ctx.deps.state.findings),
            "has_report": ctx.deps.state.final_report is not None,
        }

    return agent
