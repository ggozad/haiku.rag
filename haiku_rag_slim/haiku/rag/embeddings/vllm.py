"""Multimodal embedder backed by a vLLM OpenAI-compatible HTTP server.

vLLM's ``/v1/embeddings`` endpoint is a superset of OpenAI's:

- Text inputs use the standard ``input: list[str]`` field — one HTTP call
  returns N embeddings.
- Image inputs use a ``messages`` array carrying an ``image_url`` content
  part with a base64 data URI. One image per HTTP call.

Models like ``Qwen/Qwen3-VL-Embedding-8B`` and ``jinaai/jina-embeddings-v4``
ship with chat templates that map both shapes into a shared vector space.
"""

import logging
from typing import TYPE_CHECKING, Any

import httpx

from haiku.rag.embeddings import EmbedderWrapper, _to_data_uri

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Below vLLM's 5s server keep-alive (VLLM_HTTP_TIMEOUT_KEEP_ALIVE).
KEEPALIVE_EXPIRY = 2.0

# Raised when the connection dies under a sent request. Embedding is
# idempotent, so these are retried once.
_DROPPED_CONNECTION_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


class VLLMMultimodalEmbedder(EmbedderWrapper):
    # Subclasses serving another endpoint override these; errors quote them.
    _service_name = "vLLM"
    _connect_hint = "Ensure the service is running."

    def __init__(
        self,
        model_name: str,
        vector_dim: int,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        supports_images: bool = True,
    ):
        super().__init__(
            embedder=None, vector_dim=vector_dim, supports_images=supports_images
        )
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        # One client reused across every request so the connection (and its
        # name resolution) is established once and kept alive, rather than
        # rebuilt per call.
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=KEEPALIVE_EXPIRY,
            ),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(self, body: dict[str, Any]) -> httpx.Response:
        """POST ``body``, retrying once if the connection drops under it."""
        url = f"{self._base_url}/embeddings"
        try:
            return await self._client.post(url, json=body, headers=self._headers())
        except _DROPPED_CONNECTION_ERRORS as e:
            logger.warning(
                "%s at %s dropped the connection (%s: %s); retrying once",
                self._service_name,
                self._base_url,
                type(e).__name__,
                e,
            )
            return await self._client.post(url, json=body, headers=self._headers())

    async def _post(self, body: dict[str, Any]) -> list[list[float]]:
        try:
            response = await self._send(body)
            response.raise_for_status()
            payload = response.json()
        except _DROPPED_CONNECTION_ERRORS as e:
            raise ValueError(
                f"{self._service_name} at {self._base_url} dropped the "
                f"connection twice without a response. Error: {e}"
            ) from e
        except httpx.ConnectError as e:
            raise ValueError(
                f"Could not connect to {self._service_name} at {self._base_url}. "
                f"{self._connect_hint} Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ValueError(
                f"Request to {self._service_name} timed out after "
                f"{self._timeout}s. Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    f"Authentication failed against {self._service_name}. "
                    "Check the API key."
                ) from e
            raise ValueError(f"HTTP error from {self._service_name}: {e}") from e

        data = payload.get("data") or []
        if not data:
            raise ValueError(f"{self._service_name} returned no embeddings: {payload}")
        rows = [list(d["embedding"]) for d in data]
        for row in rows:
            if len(row) != self._vector_dim:
                raise ValueError(
                    f"{self._service_name} model '{self._model_name}' returned a "
                    f"{len(row)}-dimensional embedding, but "
                    f"embeddings.model.vector_dim is {self._vector_dim}. Set "
                    "vector_dim to the model's own dimension."
                )
        return rows

    async def embed_query(self, text: str) -> list[float]:
        rows = await self._post(
            {
                "model": self._model_name,
                "input": [text],
                "encoding_format": "float",
            }
        )
        return rows[0]

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._post(
            {
                "model": self._model_name,
                "input": texts,
                "encoding_format": "float",
            }
        )

    def _image_request(self, image: "bytes | PILImage.Image") -> dict[str, Any]:
        """Request body that embeds one picture."""
        return {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _to_data_uri(image)},
                        }
                    ],
                }
            ],
            "encoding_format": "float",
        }

    async def embed_image(self, image: "bytes | PILImage.Image") -> list[float]:
        if not self.supports_images:
            raise NotImplementedError(
                f"This {self._service_name} embedder is text-only. Set "
                "embeddings.model.multimodal: true to embed images."
            )
        rows = await self._post(self._image_request(image))
        return rows[0]
