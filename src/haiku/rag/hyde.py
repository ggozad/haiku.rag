from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config


async def generate_hypothetical_document(query: str) -> str:
    """Generate a hypothetical document for HyDE retrieval.

    HyDE (Hypothetical Document Embeddings) generates a hypothetical answer
    to the query using an LLM, then uses that answer for retrieval instead
    of the original query. This bridges the semantic gap between queries
    (questions) and documents (answers).

    Args:
        query: The search query.

    Returns:
        A hypothetical document that answers the query.
    """
    # Use HyDE-specific settings or fall back to QA settings
    provider = Config.HYDE_PROVIDER or Config.QA_PROVIDER
    model_name = Config.HYDE_MODEL or Config.QA_MODEL

    # Create a simple agent without any tools (no RAG/search)
    if provider == "ollama":
        model = OpenAIChatModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
        )
    elif provider == "vllm":
        model = OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=Config.VLLM_QA_BASE_URL),
        )
    else:
        model = model_name

    agent = Agent(model=model)

    # Use custom prompt if configured, otherwise use default
    if Config.HYDE_PROMPT:
        prompt = Config.HYDE_PROMPT.format(query=query)
    else:
        # Default prompt: shorter, more focused
        prompt = f"""Answer this question concisely as if from technical documentation.
Use specific terminology. Keep it brief (2-3 sentences).

Question: {query}"""

    result = await agent.run(prompt)
    return result.output
