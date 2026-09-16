import os
from typing import TYPE_CHECKING, Any

from haiku.rag.embeddings import _to_data_uri
from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

if TYPE_CHECKING:
    from PIL import Image as PILImage

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenRouterEmbedder(VLLMMultimodalEmbedder):
    """Embedder for OpenRouter's ``/v1/embeddings``.

    Text is the standard OpenAI ``input`` array of strings. Images wrap content
    parts in an ``input`` element, where vLLM uses a top-level ``messages``
    array, and an ``input`` may not mix strings with content objects.
    """

    _service_name = "OpenRouter"
    _connect_hint = "Check the network and the base_url."

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
            model_name,
            vector_dim,
            base_url=base_url,
            api_key=api_key or os.environ.get(API_KEY_ENV),
            timeout=timeout,
            supports_images=supports_images,
        )

    def _image_request(self, image: "bytes | PILImage.Image") -> dict[str, Any]:
        return {
            "model": self._model_name,
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _to_data_uri(image)},
                        }
                    ]
                }
            ],
            "encoding_format": "float",
        }
