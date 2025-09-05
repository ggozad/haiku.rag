from openai import AsyncOpenAI

from haiku.rag.embeddings.base import EmbedderBase


class Embedder(EmbedderBase):
    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        client = AsyncOpenAI()
        response = await client.embeddings.create(
            model=self._model,
            input=text,
        )
        if isinstance(text, str):
            return response.data[0].embedding
        else:
            return [item.embedding for item in response.data]
