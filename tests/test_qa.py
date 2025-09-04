import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.agent import QuestionAnswerAgent

from .llm_judge import LLMJudge

OPENAI_AVAILABLE = bool(Config.OPENAI_API_KEY)
ANTHROPIC_AVAILABLE = bool(Config.ANTHROPIC_API_KEY)
VLLM_QA_AVAILABLE = bool(Config.VLLM_QA_BASE_URL)


@pytest.mark.asyncio
async def test_qa_ollama(qa_corpus: Dataset, temp_db_path):
    """Test Ollama QA with LLM judge."""
    client = HaikuRAG(temp_db_path)
    qa = QuestionAnswerAgent(client, "ollama", "qwen3")
    llm_judge = LLMJudge()

    doc = qa_corpus[1]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    question = doc["question"]
    expected_answer = doc["answer"]

    answer = await qa.answer(question)
    is_equivalent = await llm_judge.judge_answers(question, answer, expected_answer)

    assert is_equivalent, (
        f"Generated answer not equivalent to expected answer.\nQuestion: {question}\nGenerated: {answer}\nExpected: {expected_answer}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
async def test_qa_openai(qa_corpus: Dataset, temp_db_path):
    """Test OpenAI QA with LLM judge."""
    client = HaikuRAG(temp_db_path)
    qa = QuestionAnswerAgent(client, "openai", "gpt-4o-mini")
    llm_judge = LLMJudge()

    doc = qa_corpus[1]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    question = doc["question"]
    expected_answer = doc["answer"]

    answer = await qa.answer(question)
    is_equivalent = await llm_judge.judge_answers(question, answer, expected_answer)

    assert is_equivalent, (
        f"Generated answer not equivalent to expected answer.\nQuestion: {question}\nGenerated: {answer}\nExpected: {expected_answer}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic not available")
async def test_qa_anthropic(qa_corpus: Dataset, temp_db_path):
    """Test Anthropic QA with LLM judge."""
    client = HaikuRAG(temp_db_path)
    qa = QuestionAnswerAgent(client, "anthropic", "claude-3-5-haiku-20241022")
    llm_judge = LLMJudge()

    doc = qa_corpus[1]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    question = doc["question"]
    expected_answer = doc["answer"]

    answer = await qa.answer(question)
    is_equivalent = await llm_judge.judge_answers(question, answer, expected_answer)

    assert is_equivalent, (
        f"Generated answer not equivalent to expected answer.\nQuestion: {question}\nGenerated: {answer}\nExpected: {expected_answer}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLM_QA_AVAILABLE, reason="vLLM QA server not configured")
async def test_qa_vllm(qa_corpus: Dataset, temp_db_path):
    """Test vLLM QA with LLM judge."""
    client = HaikuRAG(temp_db_path)
    qa = QuestionAnswerAgent(client, "vllm", "Qwen/Qwen3-4B")
    llm_judge = LLMJudge()

    doc = qa_corpus[1]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    question = doc["question"]
    expected_answer = doc["answer"]
    answer = await qa.answer(question)
    is_equivalent = await llm_judge.judge_answers(question, answer, expected_answer)

    assert is_equivalent, (
        f"Generated answer not equivalent to expected answer.\nQuestion: {question}\nGenerated: {answer}\nExpected: {expected_answer}"
    )
