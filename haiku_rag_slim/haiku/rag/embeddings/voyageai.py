try:
    from voyageai.client import Client  # type: ignore

    from haiku.rag.config import AppConfig

    class Embedder:
        """VoyageAI embedder with explicit query/document methods."""

        def __init__(self, model: str, vector_dim: int, config: AppConfig):
            self._model = model
            self._vector_dim = vector_dim
            self._config = config

        async def embed_query(self, text: str) -> list[float]:
            """Embed a search query."""
            client = Client()
            res = client.embed(
                [text], model=self._model, input_type="query", output_dtype="float"
            )
            return res.embeddings[0]  # type: ignore[return-value]

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """Embed documents/chunks for indexing."""
            if not texts:
                return []
            client = Client()
            res = client.embed(
                texts, model=self._model, input_type="document", output_dtype="float"
            )
            return res.embeddings  # type: ignore[return-value]

except ImportError:
    pass
