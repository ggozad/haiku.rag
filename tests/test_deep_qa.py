import pytest
from pydantic_ai.models.test import TestModel

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.models import SearchAnswer
from haiku.rag.qa.deep.dependencies import DeepQAContext
from haiku.rag.qa.deep.graph import build_deep_qa_graph
from haiku.rag.qa.deep.nodes import DeepQAPlanNode
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState


@pytest.mark.asyncio
async def test_deep_qa_graph_end_to_end(monkeypatch, temp_db_path):
    """Test deep Q&A graph with mocked LLM using TestModel."""
    graph = build_deep_qa_graph()

    state = DeepQAState(
        context=DeepQAContext(
            original_question="What is haiku.rag?", use_citations=False
        ),
        max_sub_questions=3,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path)
    deps = DeepQADeps(client=client, console=None)

    # Mock get_model to return TestModel which generates valid schema-compliant data
    def test_model_factory(provider, model):
        return TestModel()

    monkeypatch.setattr("haiku.rag.graph.common.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.qa.deep.nodes.get_model", test_model_factory)

    start = DeepQAPlanNode(provider="test", model="test")
    result = await graph.run(start_node=start, state=state, deps=deps)

    # TestModel will generate valid structured output based on schemas
    assert result.output.answer is not None
    assert isinstance(result.output.answer, str)
    assert isinstance(result.output.sources, list)

    client.close()


@pytest.mark.asyncio
async def test_deep_qa_with_citations(monkeypatch, temp_db_path):
    """Test deep Q&A with citations enabled using TestModel."""
    graph = build_deep_qa_graph()

    state = DeepQAState(
        context=DeepQAContext(original_question="What is Python?", use_citations=True),
        max_sub_questions=2,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path)
    deps = DeepQADeps(client=client, console=None)

    # Mock get_model to return TestModel
    def test_model_factory(provider, model):
        return TestModel()

    monkeypatch.setattr("haiku.rag.graph.common.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.qa.deep.nodes.get_model", test_model_factory)

    start = DeepQAPlanNode(provider="test", model="test")
    result = await graph.run(start_node=start, state=state, deps=deps)

    # Verify citations flag was used
    assert state.context.use_citations is True
    assert result.output.answer is not None
    assert isinstance(result.output.sources, list)

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
