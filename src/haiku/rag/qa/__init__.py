from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.agent import QuestionAnswerAgent


def get_qa_agent(
    client: HaikuRAG,
    use_citations: bool = False,
    system_prompt: str | None = None,
) -> QuestionAnswerAgent:
    provider = Config.qa.provider
    model_name = Config.qa.model

    return QuestionAnswerAgent(
        client=client,
        provider=provider,
        model=model_name,
        use_citations=use_citations,
        system_prompt=system_prompt,
    )
