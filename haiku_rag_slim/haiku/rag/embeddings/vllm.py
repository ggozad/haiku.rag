"""Multimodal embedder backed by a vLLM OpenAI-compatible HTTP server.

vLLM's ``/v1/embeddings`` endpoint is a superset of OpenAI's: when the
request body uses a ``messages`` array (instead of ``input``), the server
treats it as a chat-style multimodal embedding request and accepts
``image_url`` content parts carrying base64 data URIs. Models like
``Qwen/Qwen3-VL-Embedding-8B`` and ``jinaai/jina-embeddings-v4`` ship with
chat templates that map both text and image inputs into a shared vector
space.
"""

import asyncio
import base64
import io
from typing import TYPE_CHECKING, Any

import httpx

from haiku.rag.embeddings import EmbedderWrapper

if TYPE_CHECKING:
    from PIL import Image as PILImage


class VLLMMultimodalEmbedder(EmbedderWrapper):
    supports_images = True

    def __init__(
        self,
        model_name: str,
        vector_dim: int,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        super().__init__(embedder=None, vector_dim=vector_dim)
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _post_messages(self, content: list[dict[str, Any]]) -> list[float]:
        body = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": content}],
            "encoding_format": "float",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
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
                f"Request to vLLM timed out after {self._timeout}s. Error: {e}"
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
        return list(data[0]["embedding"])

    async def embed_query(self, text: str) -> list[float]:
        return await self._post_messages([{"type": "text", "text": text}])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.gather(*(self.embed_query(t) for t in texts))

    async def embed_image_query(self, image: "bytes | PILImage.Image") -> list[float]:
        return await self._post_messages(
            [{"type": "image_url", "image_url": {"url": _to_data_uri(image)}}]
        )

    async def embed_images(
        self, images: list["bytes | PILImage.Image"]
    ) -> list[list[float]]:
        if not images:
            return []
        return await asyncio.gather(*(self.embed_image_query(img) for img in images))


def _to_data_uri(image: "bytes | PILImage.Image") -> str:
    """Render an image as a ``data:image/png;base64,...`` URI."""
    if isinstance(image, bytes):
        return f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}"

    from PIL import Image as PILImageModule

    if isinstance(image, PILImageModule.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return (
            f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        )

    raise TypeError(f"Unsupported image type: {type(image)!r}")
