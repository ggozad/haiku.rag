from typing import Any, cast

import pytest

from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.graph import (
    AnalyzeInsightsNode,
    DecisionNode,
    PlanNode,
    ResearchDeps,
    ResearchState,
    SearchDispatchNode,
    SynthesizeNode,
    build_research_graph,
)
from haiku.rag.research.models import (
    EvaluationResult,
    GapRecord,
    GapSeverity,
    InsightAnalysis,
    InsightRecord,
    InsightStatus,
    ResearchReport,
    SearchAnswer,
)
from haiku.rag.research.stream import stream_research_graph


@pytest.mark.asyncio
async def test_graph_end_to_end_with_patched_nodes(monkeypatch):
    graph = build_research_graph()

    state = ResearchState(
        context=ResearchContext(original_question="What is haiku.rag?"),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=2,
    )
    deps = ResearchDeps(
        client=cast(Any, None), console=None
    )  # client unused in patched nodes

    async def fake_plan_run(self, ctx) -> Any:
        ctx.state.context.sub_questions = [
            "Describe haiku.rag in one sentence",
            "List core components of haiku.rag",
        ]
        ctx.deps.emit_log("planning", ctx.state)
        return SearchDispatchNode(self.provider, self.model)

    async def fake_search_dispatch_run(self, ctx) -> Any:
        # Answer all pending questions deterministically, then move to analysis
        while ctx.state.context.sub_questions:
            q = ctx.state.context.sub_questions.pop(0)
            # pydantic BaseModel kwargs not fully typed for pyright
            ctx.state.context.add_qa_response(
                SearchAnswer(query=q, answer="A", context=["x"], sources=["s"])  # pyright: ignore[reportCallIssue]
            )
            ctx.deps.emit_log(f"answered:{q}", ctx.state)
        return AnalyzeInsightsNode(self.provider, self.model)

    async def fake_analyze_run(self, ctx) -> Any:
        analysis = InsightAnalysis(
            highlights=[
                InsightRecord(
                    summary="haiku.rag orchestrates research stages",
                    status=InsightStatus.VALIDATED,
                    supporting_sources=["s"],
                    originating_questions=["Describe haiku.rag in one sentence"],
                )
            ],
            gap_assessments=[
                GapRecord(
                    description="Need a final summary",
                    severity=GapSeverity.LOW,
                    blocking=False,
                    resolved=False,
                )
            ],
            resolved_gaps=[],
            new_questions=[],
            commentary="Insights captured for synthesis",
        )
        ctx.state.context.integrate_analysis(analysis)
        ctx.state.last_analysis = analysis
        ctx.deps.emit_log("analysis", ctx.state)
        return DecisionNode(self.provider, self.model)

    async def fake_decision_run(self, ctx) -> Any:
        ctx.state.last_eval = EvaluationResult(
            key_insights=["haiku.rag coordinates planning, search, and synthesis"],
            new_questions=[],
            gaps=["Need a final summary"],
            confidence_score=1.0,
            is_sufficient=True,
            reasoning="done",
        )
        ctx.state.iterations += 1
        ctx.deps.emit_log("decision", ctx.state)
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
    monkeypatch.setattr(AnalyzeInsightsNode, "run", fake_analyze_run, raising=False)
    monkeypatch.setattr(DecisionNode, "run", fake_decision_run, raising=False)
    monkeypatch.setattr(SynthesizeNode, "run", fake_synthesize_run, raising=False)

    start = PlanNode(provider="test", model="test")

    collected = []
    async for event in stream_research_graph(graph, start, state, deps):
        collected.append(event)
        if event.type == "report":
            report = event.report
            break
    else:  # pragma: no cover - defensive guard
        report = None

    assert isinstance(report, ResearchReport)
    assert report.title == "Haiku RAG"
    assert len(state.context.qa_responses) == 2
    assert any(evt.type == "log" for evt in collected)
