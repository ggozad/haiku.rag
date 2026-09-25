import json
from collections.abc import Sequence
from pathlib import Path

from evaluations.datasets.collection_routing import shard_of
from haiku.rag.client import DocumentImport, HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import Chunk, Document


async def split_database(
    source: Path,
    destinations: Sequence[Path],
    config: AppConfig,
    *,
    batch_size: int = 32,
) -> list[int]:
    """Copy every document of `source` into the destination its uri hashes to.

    Chunks travel with their embeddings, so no embedder runs. Each destination
    records the embedder `config` names, which must be the one the source was
    built with. Returns the document count per destination.
    """
    async with HaikuRAG(
        source, config=config, read_only=True, skip_validation=True
    ) as src:
        shards: list[list[Document]] = [[] for _ in destinations]
        for document in await src.list_documents():
            if document.uri is None:
                raise ValueError(f"document {document.id} has no uri to shard by")
            shards[shard_of(document.uri, len(destinations))].append(document)
        for shard, destination in zip(shards, destinations, strict=True):
            async with HaikuRAG(destination, config=config, create=True) as dst:
                for start in range(0, len(shard), batch_size):
                    batch = shard[start : start + batch_size]
                    await dst.import_documents(
                        [await _document_import(src, d) for d in batch]
                    )
    return [len(shard) for shard in shards]


async def _document_import(src: HaikuRAG, document: Document) -> DocumentImport:
    assert document.id is not None
    stored = await src.document_repository.get_docling_data(document.id)
    assert stored is not None
    docling = stored.get_docling_document()
    assert docling is not None
    rows = await (
        src.store.chunks_table.query().where(f"document_id = '{document.id}'").to_list()
    )
    chunks = [
        Chunk(
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            order=row["order"],
            embedding=list(row["vector"]),
        )
        for row in sorted(rows, key=lambda row: row["order"])
    ]
    return DocumentImport(
        docling,
        chunks,
        uri=document.uri,
        title=document.title,
        metadata=document.metadata,
    )
