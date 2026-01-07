import re
from io import BytesIO
from typing import TYPE_CHECKING

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig, Config
from haiku.rag.providers.docling_serve import DoclingServeClient
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

# Pattern to parse refs like "#/texts/5" or "#/tables/0"
REF_PATTERN = re.compile(r"^#/(\w+)/(\d+)$")


def _resolve_label_from_document(ref: str, document: "DoclingDocument") -> str | None:
    """Resolve the label for a doc_item ref by looking it up in the document.

    The docling-serve API only returns ref strings in doc_items, not labels.
    This function resolves actual labels from the DoclingDocument.
    See: https://github.com/docling-project/docling-serve/issues/448

    Args:
        ref: JSON pointer reference like "#/texts/5" or "#/tables/0"
        document: The DoclingDocument to look up the item in

    Returns:
        The label string if found, None otherwise
    """
    match = REF_PATTERN.match(ref)
    if not match:
        return None

    collection_name = match.group(1)
    index = int(match.group(2))

    collection = getattr(document, collection_name, None)
    if collection is None or index >= len(collection):
        return None

    item = collection[index]
    return getattr(item, "label", None)


class DoclingServeChunker(DocumentChunker):
    """Remote document chunker using docling-serve API.

    Sends DoclingDocument JSON to docling-serve for chunking. Supports both hybrid
    and hierarchical chunking strategies via remote API.

    Args:
        config: Application configuration containing docling-serve settings.
    """

    def __init__(self, config: AppConfig = Config):
        self.config = config
        self.client = DoclingServeClient(
            base_url=config.providers.docling_serve.base_url,
            api_key=config.providers.docling_serve.api_key,
        )
        self.chunker_type = config.processing.chunker_type

    def _build_chunking_data(self) -> dict[str, str]:
        """Build form data for chunking request."""
        return {
            "chunking_max_tokens": str(self.config.processing.chunk_size),
            "chunking_tokenizer": self.config.processing.chunking_tokenizer,
            "chunking_merge_peers": str(
                self.config.processing.chunking_merge_peers
            ).lower(),
            "chunking_use_markdown_tables": str(
                self.config.processing.chunking_use_markdown_tables
            ).lower(),
        }

    async def _call_chunk_api(self, document: "DoclingDocument") -> list[dict]:
        """Call docling-serve chunking API and return raw chunk data.

        Args:
            document: The DoclingDocument to be split into chunks.

        Returns:
            List of chunk dictionaries from API response.

        Raises:
            ValueError: If chunking fails or service is unavailable.
        """
        # Determine endpoint based on chunker_type
        if self.chunker_type == "hierarchical":
            endpoint = "/v1/chunk/hierarchical/file/async"
        else:
            endpoint = "/v1/chunk/hybrid/file/async"

        # Export document to JSON
        doc_json = document.model_dump_json()
        doc_bytes = doc_json.encode("utf-8")

        # Prepare multipart request with DoclingDocument JSON
        files = {"files": ("document.json", BytesIO(doc_bytes), "application/json")}
        data = self._build_chunking_data()

        result = await self.client.submit_and_poll(
            endpoint=endpoint,
            files=files,
            data=data,
            name="document",
        )

        return result.get("chunks", [])

    async def chunk(self, document: "DoclingDocument") -> list[Chunk]:
        """Split the document into chunks with metadata via docling-serve.

        Extracts structured metadata from the API response including:
        - doc_item_refs: JSON pointer references to DocItems (e.g., "#/texts/5")
        - headings: Section heading hierarchy
        - labels: Semantic labels for each doc_item
        - page_numbers: Page numbers where content appears

        Args:
            document: The DoclingDocument to be split into chunks.

        Returns:
            List of Chunk containing content and structured metadata.

        Raises:
            ValueError: If chunking fails or service is unavailable.
        """
        if document is None:
            return []

        raw_chunks = await self._call_chunk_api(document)
        result: list[Chunk] = []

        for chunk in raw_chunks:
            text = chunk.get("text", "")

            # doc_items from docling-serve is a list of ref strings like ["#/texts/1", "#/tables/0"]
            doc_items = chunk.get("doc_items", [])
            doc_item_refs: list[str] = []
            labels: list[str] = []

            for item in doc_items:
                if isinstance(item, str):
                    # docling-serve returns refs as strings directly
                    doc_item_refs.append(item)
                    # Resolve label from the document using the ref
                    label = _resolve_label_from_document(item, document)
                    if label:
                        labels.append(label)
                elif isinstance(item, dict):
                    # Handle dict format if API ever returns it
                    if "self_ref" in item:
                        doc_item_refs.append(item["self_ref"])
                    if "label" in item:
                        labels.append(item["label"])

            # Get headings directly from chunk
            headings = chunk.get("headings")

            # Get page numbers directly from chunk
            page_numbers = chunk.get("page_numbers", [])

            chunk_metadata = ChunkMetadata(
                doc_item_refs=doc_item_refs,
                headings=headings,
                labels=labels,
                page_numbers=sorted(page_numbers) if page_numbers else [],
            )
            result.append(
                Chunk(
                    content=text,
                    metadata=chunk_metadata.model_dump(),
                    order=len(result),
                )
            )

        return result
