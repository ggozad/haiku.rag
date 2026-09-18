"""Experiment metadata recorded with every eval run."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from haiku.rag.config import AppConfig
from haiku.rag.config.models import ModelConfig

# Pinned judge model. Decoupled from `config.qa.model` so a user changing
# their QA model does not inadvertently change the judge — keeps cross-run
# comparisons stable. Set `evaluations.judge` in the config to override it.
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


def code_revision(path: Path | None = None) -> dict[str, Any]:
    """The commit the running code is checked out at, and whether tracked files
    carry uncommitted changes. Untracked files do not count: `.env`, logs and
    scripts sit in a worktree without changing the code that runs. Both None
    when the code is not in a git checkout."""
    root = Path(__file__).resolve().parent if path is None else path

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout

    sha = git("rev-parse", "HEAD")
    if sha is None:
        return {"git_sha": None, "git_dirty": None}
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_sha": sha.strip(),
        "git_dirty": None if status is None else bool(status.strip()),
    }


def config_hash(config: AppConfig) -> str:
    """SHA-256 of the resolved configuration, so a trace names the config it ran."""
    dumped = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(dumped.encode()).hexdigest()


async def corpus_fingerprint(db_path: Path | None, config: AppConfig) -> dict[str, Any]:
    """Identity of the database a run reads: path, row counts and the stored
    embedder. Counts and identity are None when no database is placed or the
    path does not exist; nothing is created."""
    fingerprint: dict[str, Any] = {
        "db_path": None if db_path is None else str(db_path),
        "db_documents": None,
        "db_chunks": None,
        "db_embedder_provider": None,
        "db_embedder_model": None,
        "db_embedder_dim": None,
        "db_version": None,
        "db_written_at": None,
    }
    if db_path is None or not Path(db_path).exists():
        return fingerprint

    from haiku.rag.store.info import gather_database_info

    info = await gather_database_info(db_path, config)
    if not info.exists:
        return fingerprint
    rows = {table.name: table.num_rows for table in info.tables}
    written = [
        table.latest_version_at for table in info.tables if table.latest_version_at
    ]
    fingerprint.update(
        db_documents=rows.get("documents"),
        db_chunks=rows.get("chunks"),
        db_written_at=max(written) if written else None,
        db_embedder_provider=info.embeddings.provider,
        db_embedder_model=info.embeddings.name,
        db_embedder_dim=info.embeddings.vector_dim,
        db_version=info.stored_version,
    )
    return fingerprint


def build_experiment_metadata(
    dataset_key: str,
    test_cases: int,
    config: AppConfig,
    judge_config: ModelConfig | None = None,
    capability_config: ModelConfig | None = None,
    document_filter: str | None = None,
    pair_key: str | None = None,
) -> dict[str, Any]:
    """Build experiment metadata for Logfire tracking.

    `capability_*` is the model that ran the capability.
    """
    metadata: dict[str, Any] = {
        "dataset": dataset_key,
        "test_cases": test_cases,
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
        "qa_max_searches": config.qa.max_searches,
        "qa_max_executions": config.qa.max_executions,
        "sandbox_code_timeout": config.sandbox.code_timeout,
        "sandbox_max_output_chars": config.sandbox.max_output_chars,
        "document_filter": document_filter,
        "pair_key": pair_key,
        "config_hash": config_hash(config),
        **code_revision(),
    }
    if judge_config is not None:
        metadata.update(
            {
                "judge_provider": judge_config.provider,
                "judge_model": judge_config.name,
                "judge_temperature": judge_config.temperature,
                "judge_max_tokens": judge_config.max_tokens,
                "judge_thinking": judge_config.thinking,
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
                "capability_thinking": capability_config.thinking,
                "capability_extra_body": capability_config.extra_body,
            }
        )
    return metadata
