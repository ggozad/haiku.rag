from io import BytesIO
from typing import TYPE_CHECKING

import requests

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig, Config

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DoclingServeChunker(DocumentChunker):
    """Remote document chunker using docling-serve API.

    Sends DoclingDocument JSON to docling-serve for chunking. Supports both hybrid
    and hierarchical chunking strategies via remote API.

    Args:
        config: Application configuration containing docling-serve settings.
    """

    def __init__(self, config: AppConfig = Config):
        self.config = config
        self.base_url = config.providers.docling_serve.base_url.rstrip("/")
        self.api_key = config.providers.docling_serve.api_key
        self.timeout = config.providers.docling_serve.timeout
        self.chunker_type = config.processing.chunker_type

    async def chunk(self, document: "DoclingDocument") -> list[str]:
        """Split the document into chunks via docling-serve.

        Exports the DoclingDocument to JSON and sends it to docling-serve's chunking
        endpoint. The API will chunk the document and return the text chunks.

        Args:
            document: The DoclingDocument to be split into chunks.

        Returns:
            A list of text chunks with semantic boundaries.

        Raises:
            ValueError: If chunking fails or service is unavailable.
        """
        if document is None:
            return []

        try:
            # Determine endpoint based on chunker_type
            if self.chunker_type == "hierarchical":
                url = f"{self.base_url}/v1/chunk/hierarchical/file"
            else:
                url = f"{self.base_url}/v1/chunk/hybrid/file"

            # Export document to JSON
            doc_json = document.model_dump_json()
            doc_bytes = doc_json.encode("utf-8")

            # Prepare multipart request with DoclingDocument JSON
            files = {"files": ("document.json", BytesIO(doc_bytes), "application/json")}

            # Build form data with chunking parameters
            data = {
                "chunking_max_tokens": str(self.config.processing.chunk_size),
                "chunking_tokenizer": self.config.processing.chunking_tokenizer,
                "chunking_merge_peers": str(
                    self.config.processing.chunking_merge_peers
                ).lower(),
                "chunking_use_markdown_tables": str(
                    self.config.processing.chunking_use_markdown_tables
                ).lower(),
            }

            headers = {}
            if self.api_key:
                headers["X-Api-Key"] = self.api_key

            response = requests.post(
                url,
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            result = response.json()

            # Extract text from chunks
            chunks = result.get("chunks", [])
            return [chunk["text"] for chunk in chunks]

        except requests.exceptions.ConnectionError as e:
            raise ValueError(
                f"Could not connect to docling-serve at {self.base_url}. "
                f"Ensure the service is running and accessible. Error: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise ValueError(
                f"Request to docling-serve timed out after {self.timeout}s. "
                f"Consider increasing the timeout in configuration. Error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key configuration."
                )
            raise ValueError(f"HTTP error from docling-serve: {e}")
        except Exception as e:
            raise ValueError(f"Failed to chunk via docling-serve: {e}")
