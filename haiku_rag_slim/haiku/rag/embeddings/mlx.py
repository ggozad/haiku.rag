"""In-process multimodal embedder backed by Apple's MLX framework.

Loads a Hugging Face repo that ships an MLX-formatted weight set plus a
``load_model.py`` helper (e.g. ``jinaai/jina-embeddings-v4-mlx-8bit``).
Apple Silicon only — the underlying ``mlx`` / ``mlx-lm`` packages don't
have wheels on other platforms and are guarded by environment markers in
the ``[mlx]`` extra.
"""

import asyncio
import io
import platform
import sys
from typing import TYPE_CHECKING, Any

from haiku.rag.embeddings import EmbedderWrapper

if TYPE_CHECKING:
    from PIL import Image as PILImage


_DEFAULT_TEXT_PROMPT = "<|im_start|>user\n{text}<|im_end|>"
_DEFAULT_IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|>"
)
_DEFAULT_PROCESSOR_REPO = "jinaai/jina-embeddings-v4"


class MLXEmbedder(EmbedderWrapper):
    supports_images = True

    def __init__(
        self,
        model_name: str,
        vector_dim: int,
        processor_repo: str | None = None,
    ):
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "provider='mlx' requires Apple Silicon (macOS arm64). "
                "On other platforms use provider='vllm' against a vLLM server."
            )
        super().__init__(embedder=None, vector_dim=vector_dim)
        self._model_name = model_name
        self._processor_repo = processor_repo or _DEFAULT_PROCESSOR_REPO
        self._model: Any | None = None
        self._processor: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor

        model_dir = snapshot_download(self._model_name)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from load_model import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
            load_mlx_model,
        )

        self._model = load_mlx_model(model_dir)
        self._processor = AutoProcessor.from_pretrained(self._processor_repo)
        return self._model, self._processor

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_texts, texts)

    async def embed_image_query(self, image: "bytes | PILImage.Image") -> list[float]:
        embeddings = await self.embed_images([image])
        return embeddings[0]

    async def embed_images(
        self, images: list["bytes | PILImage.Image"]
    ) -> list[list[float]]:
        if not images:
            return []
        return await asyncio.to_thread(self._encode_images, images)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        import mlx.core as mx  # ty: ignore[unresolved-import]

        model, processor = self._ensure_loaded()
        prompts = [_DEFAULT_TEXT_PROMPT.format(text=t) for t in texts]
        inputs = processor(
            text=prompts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        embeddings = model.encode_text(
            input_ids=mx.array(inputs["input_ids"]),
            attention_mask=mx.array(inputs["attention_mask"]),
            task="retrieval",
        )
        mx.eval(embeddings)
        return [list(map(float, row)) for row in embeddings]

    def _encode_images(
        self, images: list["bytes | PILImage.Image"]
    ) -> list[list[float]]:
        import mlx.core as mx  # ty: ignore[unresolved-import]
        from PIL import Image as PILImageModule

        model, processor = self._ensure_loaded()
        pil_images = [_to_pil(img) for img in images]
        out: list[list[float]] = []
        for pil_image in pil_images:
            assert isinstance(pil_image, PILImageModule.Image)
            inputs = processor(
                text=[_DEFAULT_IMAGE_PROMPT],
                images=[pil_image],
                return_tensors="np",
                padding=True,
            )
            pixel_values = inputs["pixel_values"]
            embedding = model.encode_image(
                input_ids=mx.array(inputs["input_ids"]),
                pixel_values=mx.array(pixel_values.reshape(-1, pixel_values.shape[-1])),
                image_grid_thw=[tuple(r) for r in inputs["image_grid_thw"]],
                attention_mask=mx.array(inputs["attention_mask"]),
                task="retrieval",
            )
            mx.eval(embedding)
            out.append([float(x) for x in embedding[0]])
        return out


def _to_pil(image: "bytes | PILImage.Image") -> "PILImage.Image":
    from PIL import Image as PILImageModule

    if isinstance(image, bytes):
        return PILImageModule.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, PILImageModule.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    raise TypeError(f"Unsupported image type: {type(image)!r}")
