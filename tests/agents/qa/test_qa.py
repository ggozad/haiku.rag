import importlib.util
from pathlib import Path

import pytest
from datasets import Dataset
from evaluations.evaluators import LLMJudge

from haiku.rag.agents.qa.agent import QuestionAnswerAgent
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import ModelConfig

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_qa")


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


@pytest.mark.vcr()
async def test_qa_openai(allow_model_requests, qa_corpus: Dataset, temp_db_path):
    """Test OpenAI QA with LLM judge (VCR recorded)."""
    client = HaikuRAG(temp_db_path, create=True)
    qa = QuestionAnswerAgent(client, ModelConfig(provider="openai", name="gpt-4o-mini"))
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


@pytest.mark.vcr()
@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not installed")
async def test_qa_anthropic(allow_model_requests, qa_corpus: Dataset, temp_db_path):
    """Test Anthropic QA with LLM judge (VCR recorded)."""
    client = HaikuRAG(temp_db_path, create=True)
    qa = QuestionAnswerAgent(
        client, ModelConfig(provider="anthropic", name="claude-3-5-haiku-20241022")
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
