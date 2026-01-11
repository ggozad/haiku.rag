import base64
import asyncio
from dataclasses import dataclass

import httpx

from haiku.rag.config import AppConfig, Config


@dataclass
class MultimodalEmbedInput:
    """Input for a multimodal embedding request."""

    text: str | None = None
    image_b64: str | None = None


class MultimodalEmbedderBase:
    """Base interface for multimodal embedders (text + image)."""

    vector_dim: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_images(self, images) -> list[list[float]]:  # images: list[PIL.Image.Image]
        raise NotImplementedError


class VLLMMultimodalEmbedder(MultimodalEmbedderBase):
    """vLLM-backed multimodal embedder via OpenAI-compatible `POST /v1/embeddings`."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        vector_dim: int,
        timeout: int,
        output_dim: int | None,
        image_max_side_px: int,
        embed_batch_size: int,
    ) -> None:
        # Accept base_url with or without a trailing /v1 for convenience.
        # (Docs for OpenAI-compatible servers often say to include /v1.)
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[: -len("/v1")]
        self.base_url = cleaned
        self.model = model
        self.vector_dim = vector_dim
        self.timeout = timeout
        self.output_dim = output_dim
        self.image_max_side_px = int(image_max_side_px)
        self.embed_batch_size = int(embed_batch_size)
        self.encoding_format: str = "float"

    async def _post_openai_embeddings_input(self, inputs: list[str]) -> list[list[float]]:
        """Batch text-only embeddings via OpenAI-compatible `input=[str, ...]`."""
        payload: dict = {
            "model": self.model,
            "input": inputs,
            "encoding_format": self.encoding_format,
        }
        # Only send "dimensions" when requesting a *different* output dimension.
        # Some servers/models reject the presence of this field unless they support
        # matryoshka output; for Qwen3-VL-Embedding-2B on vLLM, even sending the
        # native dim can yield 400.
        if self.output_dim is not None and int(self.output_dim) != int(self.vector_dim):
            payload["dimensions"] = int(self.output_dim)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/v1/embeddings", json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Surface the backend error body; otherwise we lose crucial debugging context.
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<unavailable>"
                raise httpx.HTTPStatusError(
                    f"{e}. Response body: {body[:2000]}",
                    request=e.request,
                    response=e.response,
                ) from e
            data = resp.json()

        rows = data.get("data") or []
        embeddings: list[list[float]] = []
        for r in rows:
            emb = r.get("embedding")
            if emb is None:
                continue
            embeddings.append(list(emb))

        if embeddings and len(embeddings[0]) != self.vector_dim:
            raise ValueError(
                f"Unexpected embedding dimension: got {len(embeddings[0])}, "
                f"expected {self.vector_dim}"
            )
        return embeddings

    async def _post_openai_embeddings_message(
        self, inp: MultimodalEmbedInput
    ) -> list[float]:
        """Single multimodal embedding via OpenAI-compatible `messages=[...]`.

        Note: vLLM treats `messages` as a *single* conversation and returns a single
        embedding (i.e., it does not batch multiple independent inputs via `messages`).
        """
        content: list[dict] = []
        if inp.text is not None and inp.text != "":
            content.append({"type": "text", "text": inp.text})
        if inp.image_b64 is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{inp.image_b64}"},
                }
            )

        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "encoding_format": self.encoding_format,
        }
        if self.output_dim is not None and int(self.output_dim) != int(self.vector_dim):
            payload["dimensions"] = int(self.output_dim)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/v1/embeddings", json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<unavailable>"
                raise httpx.HTTPStatusError(
                    f"{e}. Response body: {body[:2000]}",
                    request=e.request,
                    response=e.response,
                ) from e
            data = resp.json()

        rows = data.get("data") or []
        if not rows or rows[0].get("embedding") is None:
            raise ValueError("vLLM returned no embedding data")

        emb = list(rows[0]["embedding"])
        if emb and len(emb) != self.vector_dim:
            raise ValueError(
                f"Unexpected embedding dimension: got {len(emb)}, expected {self.vector_dim}"
            )
        return emb

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Batch text embeddings via `input=[str,...]`.
        bs = max(1, int(self.embed_batch_size))
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            out.extend(await self._post_openai_embeddings_input(texts[i : i + bs]))
        return out

    async def embed_images(self, images) -> list[list[float]]:  # list[PIL.Image.Image]
        if not images:
            return []
        # Lazy import: Pillow is optional in slim mode.
        try:
            from PIL import Image  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Multimodal image embedding requires Pillow. "
                "Install an appropriate extra or dependency (e.g. 'pillow')."
            ) from e

        inputs: list[MultimodalEmbedInput] = []
        for img in images:
            # Encode to PNG for a stable transport format.
            import io

            # Resize guardrail: keep max side bounded to reduce payload sizes and avoid
            # backend limits. (0 disables resizing.)
            if self.image_max_side_px > 0:
                max_side = max(int(img.size[0]), int(img.size[1]))
                if max_side > self.image_max_side_px:
                    from PIL import Image as PILImage

                    scale = float(self.image_max_side_px) / float(max_side)
                    new_w = max(1, int(round(img.size[0] * scale)))
                    new_h = max(1, int(round(img.size[1] * scale)))
                    img = img.resize((new_w, new_h), resample=PILImage.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            inputs.append(MultimodalEmbedInput(image_b64=image_b64))

        # vLLM `messages=[...]` returns ONE embedding per call, so we run one request per image.
        max_in_flight = max(1, int(self.embed_batch_size))
        sem = asyncio.Semaphore(max_in_flight)

        async def one(inp: MultimodalEmbedInput) -> list[float]:
            async with sem:
                return await self._post_openai_embeddings_message(inp)

        return list(await asyncio.gather(*(one(i) for i in inputs)))


def get_multimodal_embedder(config: AppConfig = Config) -> MultimodalEmbedderBase | None:
    """Factory for multimodal embedders.

    Returns None when multimodal indexing/search is disabled.
    """
    if not config.multimodal.enabled:
        return None

    mm = config.multimodal.model
    if mm.provider == "vllm":
        emb = VLLMMultimodalEmbedder(
            base_url=mm.base_url,
            model=mm.name,
            vector_dim=int(mm.vector_dim),
            timeout=int(mm.timeout),
            output_dim=mm.dimensions,
            image_max_side_px=int(config.multimodal.image_max_side_px),
            embed_batch_size=int(config.multimodal.embed_batch_size),
        )
        emb.encoding_format = mm.encoding_format
        return emb

    raise ValueError(f"Unsupported multimodal provider: {mm.provider}")

