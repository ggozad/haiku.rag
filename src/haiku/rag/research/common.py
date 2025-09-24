from typing import TYPE_CHECKING, Any

from pydantic_ai import format_as_xml
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config
from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.models import InsightAnalysis

if TYPE_CHECKING:  # pragma: no cover
    from haiku.rag.research.state import ResearchDeps, ResearchState


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


def log(deps: "ResearchDeps", state: "ResearchState", msg: str) -> None:
    deps.emit_log(msg, state)


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
        "insights": [
            {
                "id": insight.id,
                "summary": insight.summary,
                "status": insight.status.value,
                "supporting_sources": insight.supporting_sources,
                "originating_questions": insight.originating_questions,
                "notes": insight.notes,
            }
            for insight in context.insights
        ],
        "gaps": [
            {
                "id": gap.id,
                "description": gap.description,
                "severity": gap.severity.value,
                "blocking": gap.blocking,
                "resolved": gap.resolved,
                "resolved_by": gap.resolved_by,
                "supporting_sources": gap.supporting_sources,
                "notes": gap.notes,
            }
            for gap in context.gaps
        ],
    }
    return format_as_xml(context_data, root_tag="research_context")


def format_analysis_for_prompt(
    analysis: InsightAnalysis | None,
) -> str:
    """Format the latest insight analysis as XML for prompts."""

    if analysis is None:
        return "<latest_analysis />"

    data = {
        "commentary": analysis.commentary,
        "highlights": [
            {
                "id": insight.id,
                "summary": insight.summary,
                "status": insight.status.value,
                "supporting_sources": insight.supporting_sources,
                "originating_questions": insight.originating_questions,
                "notes": insight.notes,
            }
            for insight in analysis.highlights
        ],
        "gap_assessments": [
            {
                "id": gap.id,
                "description": gap.description,
                "severity": gap.severity.value,
                "blocking": gap.blocking,
                "resolved": gap.resolved,
                "resolved_by": gap.resolved_by,
                "supporting_sources": gap.supporting_sources,
                "notes": gap.notes,
            }
            for gap in analysis.gap_assessments
        ],
        "resolved_gaps": analysis.resolved_gaps,
        "new_questions": analysis.new_questions,
    }
    return format_as_xml(data, root_tag="latest_analysis")
