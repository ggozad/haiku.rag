from haiku.rag.tools.prompts import build_tools_prompt

_PROMPT_BASE = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

You have access to a knowledge base of documents. Use your tools to search and answer questions.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. NEVER call the same tool multiple times for a single user message
3. NEVER make up information - always use tools to get facts from the knowledge base"""

_PROMPT_QA_RULES = """
4. For questions: Use the "ask" tool EXACTLY ONCE - it automatically uses prior conversation context"""

_PROMPT_SEARCH_RULES = """
5. For searches: Use the "search" tool EXACTLY ONCE - it handles multi-query expansion internally"""

_PROMPT_SEARCH_OUTPUT = """
After calling search, copy the ENTIRE tool response to your output INCLUDING the content snippets. Do NOT shorten, summarize, or omit any part of the results."""

_PROMPT_CLOSING = """
Be friendly and conversational."""

_PROMPT_QA_CLOSING = (
    """ When you use the "ask" tool, summarize the key findings for the user."""
)


def build_chat_prompt(
    features: list[str],
    preamble: str | None = None,
) -> str:
    """Build a chat system prompt from the given feature list.

    Each feature adds its relevant tool guidance to the prompt.
    The base identity, critical rules, and closing are always included.

    Args:
        features: List of feature names (e.g., ["search", "documents", "qa"]).
        preamble: Optional custom identity/rules section. When provided,
            replaces the default identity prompt. Tool guidance, feature
            rules, and closing are still appended.

    Returns:
        The composed system prompt string.
    """
    parts = [preamble if preamble is not None else _PROMPT_BASE]

    # Add feature-specific critical rules
    if "qa" in features:
        parts.append(_PROMPT_QA_RULES)
    if "search" in features:
        parts.append(_PROMPT_SEARCH_RULES)

    # Tool guidance (reusable across agents)
    tools_prompt = build_tools_prompt(features)
    if tools_prompt:
        parts.append(tools_prompt)

    # Chat-specific search output rule
    if "search" in features:
        parts.append(_PROMPT_SEARCH_OUTPUT)

    parts.append(_PROMPT_CLOSING)
    if "qa" in features:
        parts.append(_PROMPT_QA_CLOSING)

    return "".join(parts)


CHAT_SYSTEM_PROMPT = build_chat_prompt(["search", "documents", "qa"])

SESSION_SUMMARY_PROMPT = """You are a session summarizer. Given a conversation history of Q&A pairs (and optionally existing context), produce a structured summary that captures key information for future context.

If a "Current Context" section is provided at the start of the input, incorporate that context into your summary. This might be initial background context from the user or a previous summary - build upon it rather than discard it.

Your summary should be concise (aim for 500-1500 tokens) and include:

1. **Key Facts Established** - Specific facts, data, or conclusions learned during the conversation
2. **Documents Referenced** - Documents or sources that were cited, with brief notes on what they contain
3. **Current Focus** - What topic or question thread the user is currently exploring

Rules:
- Extract only high-signal information that would help answer follow-up questions
- When building on existing context, merge new information with prior context
- Omit small talk, greetings, or low-confidence answers
- Use bullet points for clarity
- Keep technical details but compress verbose explanations
- Preserve document names/titles when mentioned in sources

Output the summary directly in markdown format. Do not include meta-commentary about the summary itself."""
