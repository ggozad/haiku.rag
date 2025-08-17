from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.agent import QuestionAnswerAgent


def get_qa_agent(client: HaikuRAG, use_citations: bool = False) -> QuestionAnswerAgent:
    provider = Config.QA_PROVIDER
    model_name = Config.QA_MODEL

    return QuestionAnswerAgent(
        client=client,
        provider=provider,
        model=model_name,
        use_citations=use_citations,
    )
