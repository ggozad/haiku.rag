from typing import overload

from haiku.rag.config import AppConfig, Config


class EmbedderBase:
    _model: str = Config.embeddings.model
    _vector_dim: int = Config.embeddings.vector_dim
    _config: AppConfig = Config

    def __init__(self, model: str, vector_dim: int, config: AppConfig = Config):
        self._model = model
        self._vector_dim = vector_dim
        self._config = config

    @overload
    async def embed(self, text: str) -> list[float]: ...

    @overload
    async def embed(self, text: list[str]) -> list[list[float]]: ...

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        raise NotImplementedError(
            "Embedder is an abstract class. Please implement the embed method in a subclass."
        )
