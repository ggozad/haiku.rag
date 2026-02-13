from pathlib import Path

import pytest
from datasets import Dataset
from evaluations.evaluators import LLMJudge

from haiku.rag.agents.qa.agent import QuestionAnswerAgent
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.config.models import ModelConfig


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_qa")


def test_get_qa_agent_factory(temp_db_path):
    """Test get_qa_agent factory function creates a properly configured agent."""
    from haiku.rag.agents.qa import get_qa_agent

    client = HaikuRAG(temp_db_path, create=True)
    agent = get_qa_agent(client, Config)

    assert agent is not None
    assert isinstance(agent, QuestionAnswerAgent)
    # Verify internal client is set correctly
    assert agent._client is client

    client.close()


def test_get_qa_agent_with_custom_prompt(temp_db_path):
    """Test get_qa_agent factory with custom system prompt."""
    from haiku.rag.agents.qa import get_qa_agent

    client = HaikuRAG(temp_db_path, create=True)
    custom_prompt = "You are a custom QA assistant."
    agent = get_qa_agent(client, Config, system_prompt=custom_prompt)

    assert agent is not None
    assert isinstance(agent, QuestionAnswerAgent)
    assert agent._system_prompt == custom_prompt

    client.close()


@pytest.mark.vcr()
async def test_qa_ollama(allow_model_requests, qa_corpus: Dataset, temp_db_path):
    """Test Ollama QA with LLM judge (VCR recorded)."""
    client = HaikuRAG(temp_db_path, create=True)
    qa = QuestionAnswerAgent(
        client, ModelConfig(provider="ollama", name="gpt-oss", enable_thinking=True)
    )
    llm_judge = LLMJudge()

    doc = qa_corpus[1]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    question = doc["question"]
    expected_answer = doc["answer"]

    answer, _ = await qa.answer(question)
    is_equivalent = await llm_judge.judge_answers(question, answer, expected_answer)

    assert is_equivalent, (
        f"Generated answer not equivalent to expected answer.\nQuestion: {question}\nGenerated: {answer}\nExpected: {expected_answer}"
    )
