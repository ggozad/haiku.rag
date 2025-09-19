from typing import Any, cast

import pytest

from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.graph import (
    EvaluateNode,
    PlanNode,
    ResearchDeps,
    ResearchState,
    SearchDispatchNode,
    SynthesizeNode,
    build_research_graph,
)
from haiku.rag.research.models import EvaluationResult, ResearchReport, SearchAnswer


@pytest.mark.asyncio
async def test_graph_end_to_end_with_patched_nodes(monkeypatch):
    graph = build_research_graph()

    state = ResearchState(
        question="What is haiku.rag?",
        context=ResearchContext(original_question="What is haiku.rag?"),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=2,
    )
    deps = ResearchDeps(
        client=cast(Any, None), console=None
    )  # client unused in patched nodes

    async def fake_plan_run(self, ctx) -> Any:
        ctx.state.sub_questions = [
            "Describe haiku.rag in one sentence",
            "List core components of haiku.rag",
        ]
        return SearchDispatchNode(self.provider, self.model)

    async def fake_search_dispatch_run(self, ctx) -> Any:
        # Answer all pending questions deterministically, then move to evaluation
        while ctx.state.sub_questions:
            q = ctx.state.sub_questions.pop(0)
            # pydantic BaseModel kwargs not fully typed for pyright
            ctx.state.context.add_qa_response(
                SearchAnswer(query=q, answer="A", context=["x"], sources=["s"])  # pyright: ignore[reportCallIssue]
            )
        return EvaluateNode(self.provider, self.model)

    async def fake_evaluate_run(self, ctx) -> Any:
        ctx.state.last_eval = EvaluationResult(
            key_insights=["ok"],
            new_questions=[],
            confidence_score=1.0,
            is_sufficient=True,
            reasoning="done",
        )
        ctx.state.iterations += 1
        return SynthesizeNode(self.provider, self.model)

    async def fake_synthesize_run(self, ctx) -> Any:
        report = ResearchReport(
            title="Haiku RAG",
            executive_summary="...",
            main_findings=["f1"],
            conclusions=["c1"],
            limitations=[],
            recommendations=[],
            sources_summary="s",
        )
        from pydantic_graph import End

        return End(report)

    monkeypatch.setattr(PlanNode, "run", fake_plan_run, raising=False)
    monkeypatch.setattr(
        SearchDispatchNode, "run", fake_search_dispatch_run, raising=False
    )
    monkeypatch.setattr(EvaluateNode, "run", fake_evaluate_run, raising=False)
    monkeypatch.setattr(SynthesizeNode, "run", fake_synthesize_run, raising=False)

    start = PlanNode(provider="test", model="test")

    result = await graph.run(start, state=state, deps=deps)
    report = result.output

    assert isinstance(report, ResearchReport)
    assert report.title == "Haiku RAG"
    assert len(state.context.qa_responses) == 2
