import pytest
from pydantic_ai.models.test import TestModel

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.common.models import SearchAnswer
from haiku.rag.graph.deep_qa.dependencies import DeepQAContext
from haiku.rag.graph.deep_qa.graph import build_deep_qa_graph
from haiku.rag.graph.deep_qa.state import DeepQADeps, DeepQAState


@pytest.mark.asyncio
async def test_deep_qa_graph_end_to_end(monkeypatch, temp_db_path):
    """Test deep Q&A graph with mocked LLM using TestModel."""

    # Mock get_model to return TestModel which generates valid schema-compliant data
    def test_model_factory(provider, model, config=None):
        return TestModel()

    # Patch all locations where get_model is imported
    monkeypatch.setattr("haiku.rag.utils.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.common.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.common.nodes.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.deep_qa.graph.get_model", test_model_factory)

    graph = build_deep_qa_graph()

    state = DeepQAState(
        context=DeepQAContext(
            original_question="What is haiku.rag?", use_citations=False
        ),
        max_sub_questions=3,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path, create=True)
    deps = DeepQADeps(client=client)

    result = await graph.run(state=state, deps=deps)

    # TestModel will generate valid structured output based on schemas
    assert result.answer is not None
    assert isinstance(result.answer, str)
    assert isinstance(result.sources, list)

    client.close()


@pytest.mark.asyncio
async def test_deep_qa_with_citations(monkeypatch, temp_db_path):
    """Test deep Q&A with citations enabled using TestModel."""

    # Mock get_model to return TestModel
    def test_model_factory(provider, model, config=None):
        return TestModel()

    # Patch all locations where get_model is imported
    monkeypatch.setattr("haiku.rag.utils.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.common.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.common.nodes.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.deep_qa.graph.get_model", test_model_factory)

    graph = build_deep_qa_graph()

    state = DeepQAState(
        context=DeepQAContext(original_question="What is Python?", use_citations=True),
        max_sub_questions=2,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path, create=True)
    deps = DeepQADeps(client=client)

    result = await graph.run(state=state, deps=deps)

    # Verify citations flag was used
    assert state.context.use_citations is True
    assert result.answer is not None
    assert isinstance(result.sources, list)

    client.close()


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
