from typing import Any

from pydantic_ai import format_as_xml
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config
from haiku.rag.research.dependencies import ResearchContext


def get_model(provider: str, model: str) -> Any:
    if provider == "ollama":
        return OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
        )
    elif provider == "vllm":
        return OpenAIChatModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{Config.VLLM_RESEARCH_BASE_URL or Config.VLLM_QA_BASE_URL}/v1",
                api_key="none",
            ),
        )
    else:
        return f"{provider}:{model}"


def log(console, msg: str) -> None:
    if console:
        console.print(msg)


def format_context_for_prompt(context: ResearchContext) -> str:
    """Format the research context as XML for inclusion in prompts."""

    context_data = {
        "original_question": context.original_question,
        "unanswered_questions": context.sub_questions,
        "qa_responses": [
            {
                "question": qa.query,
                "answer": qa.answer,
                "context_snippets": qa.context,
                "sources": qa.sources,  # pyright: ignore[reportAttributeAccessIssue]
            }
            for qa in context.qa_responses
        ],
        "insights": context.insights,
        "gaps": context.gaps,
    }
    return format_as_xml(context_data, root_tag="research_context")
