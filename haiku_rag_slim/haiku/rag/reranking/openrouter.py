import os

from haiku.rag.reranking.vllm import VLLMReranker
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.utils.images import image_data_uri

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenRouterReranker(VLLMReranker):
    """Reranker for OpenRouter's ``/v1/rerank``.

    A document is ``{"text": ..., "image": ...}`` with at least one of the two,
    where vLLM takes a ``content`` array of parts.
    ``nvidia/llama-nemotron-rerank-vl-1b-v2:free`` is the only model that takes
    images.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model,
            base_url or BASE_URL,
            api_key=api_key or os.environ.get(API_KEY_ENV),
        )

    def _document(self, chunk: Chunk) -> str | dict:
        data = chunk._picture_data
        if data is None:
            return chunk.content

        document: dict = {"image": image_data_uri(data)}
        if chunk.content:
            document["text"] = chunk.content
        return document
