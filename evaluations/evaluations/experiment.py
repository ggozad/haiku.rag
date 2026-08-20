"""Experiment metadata recorded with every eval run."""

from typing import TYPE_CHECKING, Any

from haiku.rag.config import AppConfig
from haiku.rag.config.models import ModelConfig

if TYPE_CHECKING:
    from evaluations.qa import Target

# Pinned judge model. Decoupled from `config.qa.model` so a user changing
# their QA model does not inadvertently change the judge — keeps cross-run
# comparisons stable. Override per-run with `--judge-model provider:name`.
#
# Sampling follows Qwen's recommendation for thinking mode; its model cards
# forbid greedy decoding. Only the keys ollama honours are set: it silently
# ignores `top_k`, `min_p` and `chat_template_kwargs`. The vLLM reference
# configs under `evaluations/configs/` carry those too, plus
# `reasoning_effort`, which qwen3.8 reads from `chat_template_kwargs`.
DEFAULT_JUDGE_MODEL = ModelConfig(
    provider="ollama",
    name="qwen3.8",
    temperature=0.6,
    max_tokens=16384,
    extra_body={"top_p": 0.95},
)


def build_experiment_metadata(
    dataset_key: str,
    test_cases: int,
    config: AppConfig,
    judge_config: ModelConfig | None = None,
    target: "Target" = "rag-capability",
    capability_config: ModelConfig | None = None,
    document_filter: str | None = None,
) -> dict[str, Any]:
    """Build experiment metadata for Logfire tracking."""
    metadata: dict[str, Any] = {
        "dataset": dataset_key,
        "test_cases": test_cases,
        "target": target,
        "embedder_provider": config.embeddings.model.provider,
        "embedder_model": config.embeddings.model.name,
        "embedder_dim": config.embeddings.model.vector_dim,
        "chunk_size": config.processing.chunk_size,
        "search_limit": config.search.limit,
        "max_context_chars": config.search.max_context_chars,
        "rerank_provider": config.reranking.model.provider
        if config.reranking.model
        else None,
        "rerank_model": config.reranking.model.name if config.reranking.model else None,
        "qa_provider": config.qa.model.provider,
        "qa_model": config.qa.model.name,
        "qa_temperature": config.qa.model.temperature,
        "qa_max_tokens": config.qa.model.max_tokens,
        "qa_enable_thinking": config.qa.model.enable_thinking,
        "qa_extra_body": config.qa.model.extra_body,
        "qa_max_searches": config.qa.max_searches,
        "document_filter": document_filter,
    }
    if judge_config is not None:
        metadata.update(
            {
                "judge_provider": judge_config.provider,
                "judge_model": judge_config.name,
                "judge_temperature": judge_config.temperature,
                "judge_max_tokens": judge_config.max_tokens,
                "judge_enable_thinking": judge_config.enable_thinking,
                # Sampling and thinking reach vLLM through extra_body, so
                # without it a trace cannot tell which judge settings ran.
                "judge_extra_body": judge_config.extra_body,
            }
        )
    if capability_config is not None:
        metadata.update(
            {
                "capability_provider": capability_config.provider,
                "capability_model": capability_config.name,
                "capability_temperature": capability_config.temperature,
                "capability_max_tokens": capability_config.max_tokens,
                "capability_enable_thinking": capability_config.enable_thinking,
                "capability_extra_body": capability_config.extra_body,
            }
        )
    return metadata
