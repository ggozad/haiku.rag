from typing import Any, cast

import pytest

from haiku.rag.graph.models import SearchAnswer
from haiku.rag.qa.deep.dependencies import DeepQAContext
from haiku.rag.qa.deep.graph import build_deep_qa_graph
from haiku.rag.qa.deep.models import DeepAnswer
from haiku.rag.qa.deep.nodes import (
    DeepDecisionNode,
    DeepPlanNode,
    DeepSearchDispatchNode,
    DeepSynthesizeNode,
)
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState


@pytest.mark.asyncio
async def test_deep_qa_graph_end_to_end(monkeypatch):
    graph = build_deep_qa_graph()

    state = DeepQAState(
        context=DeepQAContext(
            original_question="What is haiku.rag?", use_citations=False
        ),
        max_sub_questions=3,
    )
    deps = DeepQADeps(client=cast(Any, None), console=None)

    async def fake_plan_run(self, ctx) -> Any:
        ctx.state.context.sub_questions = [
            "Describe haiku.rag in one sentence",
            "List core components of haiku.rag",
        ]
        return DeepSearchDispatchNode(self.provider, self.model)

    async def fake_search_dispatch_run(self, ctx) -> Any:
        if not ctx.state.context.sub_questions:
            return DeepDecisionNode(self.provider, self.model)

        batch = ctx.state.context.sub_questions[:]
        ctx.state.context.sub_questions.clear()

        for question in batch:
            ctx.state.context.add_qa_response(
                SearchAnswer(
                    query=question,
                    answer=f"Answer to: {question}",
                    context=["Context snippet"],
                    sources=["test.md"],
                )
            )
        return DeepSearchDispatchNode(self.provider, self.model)

    async def fake_decision_run(self, ctx) -> Any:
        ctx.state.iterations += 1
        return DeepSynthesizeNode(self.provider, self.model)

    async def fake_synthesize_run(self, ctx) -> Any:
        from pydantic_graph import End

        return End(
            DeepAnswer(
                answer="haiku.rag is a RAG system with components A, B, C.",
                sources=["test.md"],
            )
        )

    monkeypatch.setattr(DeepPlanNode, "run", fake_plan_run)
    monkeypatch.setattr(DeepSearchDispatchNode, "run", fake_search_dispatch_run)
    monkeypatch.setattr(DeepDecisionNode, "run", fake_decision_run)
    monkeypatch.setattr(DeepSynthesizeNode, "run", fake_synthesize_run)

    start = DeepPlanNode(provider="ollama", model="test")
    result = await graph.run(start_node=start, state=state, deps=deps)

    assert result.output.answer == "haiku.rag is a RAG system with components A, B, C."
    assert result.output.sources == ["test.md"]
    assert len(state.context.qa_responses) == 2


@pytest.mark.asyncio
async def test_deep_qa_with_citations(monkeypatch):
    graph = build_deep_qa_graph()

    state = DeepQAState(
        context=DeepQAContext(original_question="What is Python?", use_citations=True),
        max_sub_questions=2,
    )
    deps = DeepQADeps(client=cast(Any, None), console=None)

    async def fake_plan_run(self, ctx) -> Any:
        ctx.state.context.sub_questions = ["What is Python used for?"]
        return DeepSearchDispatchNode(self.provider, self.model)

    async def fake_search_dispatch_run(self, ctx) -> Any:
        if not ctx.state.context.sub_questions:
            return DeepDecisionNode(self.provider, self.model)

        batch = ctx.state.context.sub_questions[:]
        ctx.state.context.sub_questions.clear()

        for question in batch:
            ctx.state.context.add_qa_response(
                SearchAnswer(
                    query=question,
                    answer="Python is used for web development and data science.",
                    context=["Python snippet"],
                    sources=["python.md"],
                )
            )
        return DeepSearchDispatchNode(self.provider, self.model)

    async def fake_decision_run(self, ctx) -> Any:
        ctx.state.iterations += 1
        return DeepSynthesizeNode(self.provider, self.model)

    async def fake_synthesize_run(self, ctx) -> Any:
        from pydantic_graph import End

        return End(
            DeepAnswer(
                answer="Python is a programming language [python.md].",
                sources=["python.md"],
            )
        )

    monkeypatch.setattr(DeepPlanNode, "run", fake_plan_run)
    monkeypatch.setattr(DeepSearchDispatchNode, "run", fake_search_dispatch_run)
    monkeypatch.setattr(DeepDecisionNode, "run", fake_decision_run)
    monkeypatch.setattr(DeepSynthesizeNode, "run", fake_synthesize_run)

    start = DeepPlanNode(provider="ollama", model="test")
    result = await graph.run(start_node=start, state=state, deps=deps)

    assert "[python.md]" in result.output.answer
    assert state.context.use_citations is True


@pytest.mark.asyncio
async def test_deep_qa_context_operations():
    context = DeepQAContext(original_question="Test question?")

    assert context.original_question == "Test question?"
    assert context.sub_questions == []
    assert context.qa_responses == []
    assert context.use_citations is False

    context.sub_questions = ["Sub Q1", "Sub Q2"]
    assert len(context.sub_questions) == 2

    qa = SearchAnswer(
        query="Sub Q1",
        answer="Answer 1",
        context=["Context 1"],
        sources=["source1.md"],
    )
    context.add_qa_response(qa)
    assert len(context.qa_responses) == 1
    assert context.qa_responses[0].query == "Sub Q1"


def test_deep_qa_state_initialization():
    context = DeepQAContext(original_question="Test?")
    state = DeepQAState(context=context, max_sub_questions=5)

    assert state.context.original_question == "Test?"
    assert state.max_sub_questions == 5
