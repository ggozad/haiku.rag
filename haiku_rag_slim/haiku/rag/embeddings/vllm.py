"""Multimodal embedder backed by a vLLM OpenAI-compatible HTTP server.

vLLM's ``/v1/embeddings`` endpoint is a superset of OpenAI's:

- Text inputs use the standard ``input: list[str]`` field — one HTTP call
  returns N embeddings.
- Image inputs use a ``messages`` array carrying an ``image_url`` content
  part with a base64 data URI. One image per HTTP call.

Models like ``Qwen/Qwen3-VL-Embedding-8B`` and ``jinaai/jina-embeddings-v4``
ship with chat templates that map both shapes into a shared vector space.
"""

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from haiku.rag.config import EmbeddingHTTPConfig
from haiku.rag.embeddings import EmbedderWrapper, _to_data_uri, build_http_client

if TYPE_CHECKING:
    from PIL import Image as PILImage


class VLLMMultimodalEmbedder(EmbedderWrapper):
    def __init__(
        self,
        model_name: str,
        vector_dim: int,
        base_url: str,
        api_key: str | None = None,
        http: EmbeddingHTTPConfig | None = None,
        supports_images: bool = True,
    ):
        super().__init__(
            embedder=None, vector_dim=vector_dim, supports_images=supports_images
        )
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        # Connection-pool + timeout tuning from config (defaults preserve the
        # historical 60s / 16-connection behavior when constructed directly).
        self._http = http or EmbeddingHTTPConfig()
        # One pooled client reused across every request (text, query, image) so
        # a connection — and its name resolution — is established once and kept
        # warm, rather than a fresh connect per call. Built lazily inside a
        # running loop; the lock ensures concurrent first-callers create only
        # one.
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = build_http_client(self._http)
        return self._client

    async def aclose(self) -> None:
        """Close the pooled HTTP client. Idempotent and safe to call on
        teardown even when no request was ever made (the client is lazy)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, body: dict[str, Any]) -> list[list[float]]:
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self._base_url}/embeddings",
                json=body,
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError as e:
            raise ValueError(
                f"Could not connect to vLLM at {self._base_url}. "
                f"Ensure the service is running. Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ValueError(
                f"Request to vLLM timed out after {self._http.timeout_s}s. "
                f"Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed against vLLM. Check the API key."
                ) from e
            raise ValueError(f"HTTP error from vLLM: {e}") from e

        data = payload.get("data") or []
        if not data:
            raise ValueError(f"vLLM returned no embeddings: {payload}")
        return [list(d["embedding"]) for d in data]

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

    async def embed_image(self, image: "bytes | PILImage.Image") -> list[float]:
        if not self.supports_images:
            raise NotImplementedError(
                "This vLLM embedder is text-only. Set "
                "embeddings.model.multimodal: true to embed images."
            )
        rows = await self._post(
            {
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
        )
        return rows[0]
