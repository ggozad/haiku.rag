import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx

from haiku.rag.config import AppConfig


@dataclass
class DownloadProgress:
    """Progress event for model downloads."""

    model: str
    status: str
    completed: int = 0
    total: int = 0
    digest: str = ""


async def download_models(
    config: AppConfig,
) -> AsyncGenerator[DownloadProgress, None]:
    """Download required models per config, yielding progress events.

    Yields DownloadProgress events for:
    - Docling models
    - HuggingFace tokenizer
    - Sentence-transformers embedder (if configured)
    - HuggingFace reranker models (mxbai, jina-local)
    - Ollama models
    """
    # Docling models
    try:
        from docling.utils.model_downloader import download_models

        yield DownloadProgress(model="docling", status="start")
        await asyncio.to_thread(download_models)
        yield DownloadProgress(model="docling", status="done")
    except ImportError:
        pass

    # HuggingFace tokenizer
    from transformers import AutoTokenizer

    tokenizer_name = config.processing.chunking_tokenizer
    yield DownloadProgress(model=tokenizer_name, status="start")
    await asyncio.to_thread(AutoTokenizer.from_pretrained, tokenizer_name)
    yield DownloadProgress(model=tokenizer_name, status="done")

    # Sentence-transformers embedder
    if config.embeddings.model.provider == "sentence-transformers":  # pragma: no cover
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                SentenceTransformer,
            )

            model_name = config.embeddings.model.name
            yield DownloadProgress(model=model_name, status="start")
            await asyncio.to_thread(SentenceTransformer, model_name)
            yield DownloadProgress(model=model_name, status="done")
        except ImportError:
            pass

    # HuggingFace reranker models
    if config.reranking.model:  # pragma: no cover
        provider = config.reranking.model.provider
        model_name = config.reranking.model.name

        if provider == "mxbai":
            try:
                from mxbai_rerank import MxbaiRerankV2

                yield DownloadProgress(model=model_name, status="start")
                await asyncio.to_thread(
                    MxbaiRerankV2, model_name, disable_transformers_warnings=True
                )
                yield DownloadProgress(model=model_name, status="done")
            except ImportError:
                pass

        elif provider == "jina-local":
            try:
                from transformers import AutoModel

                yield DownloadProgress(model=model_name, status="start")
                await asyncio.to_thread(
                    AutoModel.from_pretrained,
                    model_name,
                    trust_remote_code=True,
                )
                yield DownloadProgress(model=model_name, status="done")
            except ImportError:
                pass

    # Collect Ollama models from config
    required_models: set[str] = set()
    if config.embeddings.model.provider == "ollama":
        required_models.add(config.embeddings.model.name)
    if config.qa.model.provider == "ollama":
        required_models.add(config.qa.model.name)
    if config.research.model.provider == "ollama":
        required_models.add(config.research.model.name)
    if config.reranking.model and config.reranking.model.provider == "ollama":
        required_models.add(config.reranking.model.name)
    pic_desc = config.processing.conversion_options.picture_description
    if (
        config.processing.pictures == "description"
        and pic_desc.model.provider == "ollama"
    ):
        required_models.add(pic_desc.model.name)
    if (
        config.processing.auto_title
        and config.processing.title_model.provider == "ollama"
    ):
        required_models.add(config.processing.title_model.name)

    if not required_models:
        return

    base_url = config.providers.ollama.base_url

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            for model in sorted(required_models):
                yield DownloadProgress(model=model, status="pulling")

                async with client.stream(
                    "POST", f"{base_url}/api/pull", json={"model": model}
                ) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            digest = data.get("digest", "")

                            if digest and "total" in data:
                                yield DownloadProgress(
                                    model=model,
                                    status="downloading",
                                    total=data.get("total", 0),
                                    completed=data.get("completed", 0),
                                    digest=digest,
                                )
                            elif status:
                                yield DownloadProgress(model=model, status=status)
                        except json.JSONDecodeError:
                            pass

                yield DownloadProgress(model=model, status="done")
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is Ollama running? Start it with 'ollama serve'."
        )
