"""Multimodal embedder backed by VoyageAI's ``multimodal_embed`` endpoint.

``voyage-multimodal-3`` maps text and images into a shared vector space. Text is
embedded as single-element content lists; images are passed as ``PIL.Image``
objects (the SDK accepts them directly). The API key is read from the
environment (``VOYAGE_API_KEY``) like the text-only Voyage path.
"""

import io
from typing import TYPE_CHECKING

from haiku.rag.embeddings import EmbedderWrapper

if TYPE_CHECKING:
    from PIL import Image as PILImage


class VoyageMultimodalEmbedder(EmbedderWrapper):
    def __init__(
        self,
        model_name: str,
        vector_dim: int,
        api_key: str | None = None,
    ):
        super().__init__(embedder=None, vector_dim=vector_dim, supports_images=True)
        import voyageai

        self._model_name = model_name
        self._client = voyageai.AsyncClient(api_key=api_key)

    async def embed_query(self, text: str) -> list[float]:
        result = await self._client.multimodal_embed(
            inputs=[[text]],
            model=self._model_name,
            input_type="query",
            output_dimension=self._vector_dim,
        )
        return list(result.embeddings[0])

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.multimodal_embed(
            inputs=[[text] for text in texts],
            model=self._model_name,
            input_type="document",
            output_dimension=self._vector_dim,
        )
        return [list(e) for e in result.embeddings]

    async def embed_image(self, image: "bytes | PILImage.Image") -> list[float]:
        result = await self._client.multimodal_embed(
            inputs=[[_to_pil(image)]],
            model=self._model_name,
            input_type="document",
            output_dimension=self._vector_dim,
        )
        return list(result.embeddings[0])


def _to_pil(image: "bytes | PILImage.Image") -> "PILImage.Image":
    from PIL import Image as PILImageModule

    if isinstance(image, bytes):
        return PILImageModule.open(io.BytesIO(image))
    if isinstance(image, PILImageModule.Image):
        return image
    raise TypeError(f"Unsupported image type: {type(image)!r}")
