from typing import overload

from openai import AsyncOpenAI

from haiku.rag.embeddings.base import EmbedderBase


class Embedder(EmbedderBase):
    @overload
    async def embed(self, text: str) -> list[float]: ...

    @overload
    async def embed(self, text: list[str]) -> list[list[float]]: ...

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        client = AsyncOpenAI()
        if not text:
            return []
        response = await client.embeddings.create(
            model=self._model,
            input=text,
        )
        if isinstance(text, str):
            return response.data[0].embedding
        else:
            return [item.embedding for item in response.data]
