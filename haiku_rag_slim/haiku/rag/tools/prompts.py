_TOOL_HEADER = """

How to decide which tool to use:"""

_TOOL_DOCUMENTS = """
- "list_documents" - Use when the user wants to browse or see what documents are available (e.g., "what documents are available?", "show me the documents", "list available docs").
- "summarize_document" - Use when the user wants an overview or summary of a specific document (e.g., "summarize document X", "what does Y cover?", "give me an overview of Z").
- "get_document" - Use when the user wants the FULL content of a specific document (e.g., "get the paper about Y", "fetch 2412.00566", "show me the full document")."""

_TOOL_QA = """
- "ask" - Use for questions about topics in the knowledge base. Searches across documents and returns answers with citations. Prior answers are recalled to avoid redundant work."""

_TOOL_SEARCH = """
- "search" - Use when the user explicitly asks to search, find, or explore documents. Handles multi-query expansion internally and returns matching passages with surrounding context."""

_TOOL_ANALYSIS = """
- "analyze" - Use when the user asks for computation, data analysis, or quantitative tasks that require code execution (e.g., "calculate the average", "compare the numbers", "plot the data"). Runs Python code in a sandbox to produce results."""

_DOCUMENT_NAME_HEADER = """

IMPORTANT - When user mentions a document in search/ask:
- If user says "search in <doc>", "find in <doc>", "answer from <doc>", or "<topic> in <doc>":
  - Extract the TOPIC as `query`/`question`
  - Extract the DOCUMENT NAME as `document_name`"""

_DOCUMENT_NAME_SEARCH_EXAMPLES = """
- Examples for search:
  - "search for embeddings in the ML paper" → query="embeddings", document_name="ML paper"
  - "find transformer architecture in 2412.00566" → query="transformer architecture", document_name="2412.00566" """

_DOCUMENT_NAME_QA_EXAMPLES = """
- Examples for ask:
  - "what does the ML paper say about embeddings?" → question="what are the embedding methods?", document_name="ML paper"
  - "answer from 2412.00566 about model training" → question="how is the model trained?", document_name="2412.00566" """

_FEATURE_TOOLS: dict[str, str] = {
    "documents": _TOOL_DOCUMENTS,
    "qa": _TOOL_QA,
    "search": _TOOL_SEARCH,
    "analysis": _TOOL_ANALYSIS,
}


def build_tools_prompt(features: list[str]) -> str:
    """Build tool guidance for the given features.

    Returns prompt text describing when and how to use each tool.
    Designed to be spliced into a custom agent's system prompt.

    Args:
        features: List of feature names (e.g., ["search", "documents", "qa"]).

    Returns:
        Tool guidance prompt text.
    """
    parts: list[str] = []

    tool_sections = [_FEATURE_TOOLS[f] for f in features if f in _FEATURE_TOOLS]

    if tool_sections:
        parts.append(_TOOL_HEADER)
        parts.extend(tool_sections)

    if "search" in features or "qa" in features:
        parts.append(_DOCUMENT_NAME_HEADER)
        if "search" in features:
            parts.append(_DOCUMENT_NAME_SEARCH_EXAMPLES)
        if "qa" in features:
            parts.append(_DOCUMENT_NAME_QA_EXAMPLES)

    return "".join(parts)
