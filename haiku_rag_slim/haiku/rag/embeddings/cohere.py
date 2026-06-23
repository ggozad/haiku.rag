"""Multimodal embedder backed by Cohere's ``embed`` API (``embed-v4.0``).

``embed-v4.0`` maps text and images into a shared vector space. Text uses the
``search_document``/``search_query`` input types; images are passed as base64
data URIs with the ``image`` input type. The API key is read from the
environment (``CO_API_KEY``) like the text-only Cohere path.
"""

from typing import TYPE_CHECKING

from haiku.rag.embeddings import EmbedderWrapper, _to_data_uri

if TYPE_CHECKING:
    from PIL import Image as PILImage


class CohereMultimodalEmbedder(EmbedderWrapper):
    def __init__(
        self,
        model_name: str,
        vector_dim: int,
        api_key: str | None = None,
    ):
        super().__init__(embedder=None, vector_dim=vector_dim, supports_images=True)
        import cohere

        self._model_name = model_name
        self._client = cohere.AsyncClientV2(api_key=api_key)

    async def _embed_texts(
        self, texts: list[str], input_type: str
    ) -> list[list[float]]:
        result = await self._client.embed(
            model=self._model_name,
            input_type=input_type,
            texts=texts,
            output_dimension=self._vector_dim,
            embedding_types=["float"],
        )
        return _floats(result)

    async def embed_query(self, text: str) -> list[float]:
        rows = await self._embed_texts([text], "search_query")
        return rows[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed_texts(texts, "search_document")

    async def embed_image(self, image: "bytes | PILImage.Image") -> list[float]:
        result = await self._client.embed(
            model=self._model_name,
            input_type="image",
            images=[_to_data_uri(image)],
            output_dimension=self._vector_dim,
            embedding_types=["float"],
        )
        return _floats(result)[0]


def _floats(result: object) -> list[list[float]]:
    floats = result.embeddings.float_  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    if floats is None:
        raise ValueError("Cohere returned no float embeddings.")
    return [list(e) for e in floats]
