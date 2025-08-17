from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.agent import QuestionAnswerAgent


def get_qa_agent(client: HaikuRAG, use_citations: bool = False) -> QuestionAnswerAgent:
    provider_model = Config.QA_PROVIDER

    return QuestionAnswerAgent(
        client=client,
        provider_model=provider_model,
        use_citations=use_citations,
    )
