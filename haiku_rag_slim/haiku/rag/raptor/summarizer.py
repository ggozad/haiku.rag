from pydantic_ai import Agent

from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.raptor.prompts import CLUSTER_SUMMARY_PROMPT
from haiku.rag.utils import get_model


class ClusterSummarizer:
    """Summarizes clusters of text chunks for RAPTOR tree building."""

    def __init__(
        self,
        config: AppConfig,
        model_config: ModelConfig | None = None,
    ):
        """Initialize the summarizer.

        Args:
            config: Application configuration
            model_config: Optional model config override. If None, uses
                         config.raptor.model or falls back to config.qa.model
        """
        self._config = config

        # Determine which model to use
        if model_config is not None:
            effective_model = model_config
        elif config.raptor.model is not None:
            effective_model = config.raptor.model
        else:
            effective_model = config.qa.model

        model = get_model(effective_model, config)
        self._agent: Agent[None, str] = Agent(
            model=model,
            output_type=str,
            retries=2,
        )

    async def summarize(self, texts: list[str]) -> str:
        """Summarize a cluster of text chunks.

        Args:
            texts: List of text chunks to summarize

        Returns:
            A summary of the combined texts

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Cannot summarize empty list of texts")

        # Format chunks for the prompt
        chunks_text = "\n\n---\n\n".join(
            f"Chunk {i + 1}:\n{text}" for i, text in enumerate(texts)
        )
        prompt = CLUSTER_SUMMARY_PROMPT.format(chunks=chunks_text)

        result = await self._agent.run(prompt)
        return result.output
