try:
    from typing import overload

    from voyageai.client import Client  # type: ignore

    from haiku.rag.embeddings.base import EmbedderBase

    class Embedder(EmbedderBase):
        @overload
        async def embed(self, text: str) -> list[float]: ...

        @overload
        async def embed(self, text: list[str]) -> list[list[float]]: ...

        async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
            client = Client()
            if not text:
                return []
            if isinstance(text, str):
                res = client.embed([text], model=self._model, output_dtype="float")
                return res.embeddings[0]  # type: ignore[return-value]
            else:
                res = client.embed(text, model=self._model, output_dtype="float")
                return res.embeddings  # type: ignore[return-value]

except ImportError:
    pass
