from pydantic_ai import format_as_xml

from haiku.rag.graph.research.dependencies import ResearchContext


def format_context_for_prompt(context: ResearchContext) -> str:
    """Format the research context as XML for inclusion in prompts."""
    context_data = {
        "original_question": context.original_question,
        "unanswered_questions": context.sub_questions,
        "qa_responses": [
            {
                "question": qa.query,
                "answer": qa.answer,
                "confidence": qa.confidence,
                "sources": [
                    {
                        "document_uri": c.document_uri,
                        "document_title": c.document_title,
                        "page_numbers": c.page_numbers,
                        "headings": c.headings,
                    }
                    for c in qa.citations
                ],
            }
            for qa in context.qa_responses
        ],
    }
    return format_as_xml(context_data, root_tag="research_context")
