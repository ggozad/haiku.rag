import json
import logging
from uuid import uuid4

from haiku.rag.store.engine import Store
from haiku.rag.store.models.mm_asset import MMAsset, MMSearchResult

logger = logging.getLogger(__name__)


class MMAssetRepository:
    """Repository for multimodal asset operations."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def _require_table(self):
        if self.store.mm_assets_table is None:
            raise ValueError(
                "mm_assets table is not available. Enable config.multimodal.enabled "
                "and open the database in writable mode at least once to create it."
            )
        return self.store.mm_assets_table

    async def create(self, entity: MMAsset | list[MMAsset]) -> MMAsset | list[MMAsset]:
        """Create one or more multimodal assets.

        Assets must have embedding set before calling this method.
        """
        self.store._assert_writable()
        table = self._require_table()

        if isinstance(entity, MMAsset):
            assert entity.embedding is not None, "MMAsset must have an embedding"
            asset_id = str(uuid4())
            rec = self.store.MMAssetRecord(
                id=asset_id,
                document_id=entity.document_id,
                doc_item_ref=entity.doc_item_ref,
                item_index=entity.item_index,
                page_no=entity.page_no,
                bbox=json.dumps(entity.bbox) if entity.bbox else None,
                caption=entity.caption,
                description=entity.description,
                metadata=json.dumps(entity.metadata or {}),
                vector=entity.embedding,
            )
            table.add([rec])
            entity.id = asset_id
            return entity

        assets = entity
        if not assets:
            return []

        for a in assets:
            assert a.embedding is not None, "All MMAssets must have embeddings"

        records = []
        for a in assets:
            asset_id = str(uuid4())
            records.append(
                self.store.MMAssetRecord(
                    id=asset_id,
                    document_id=a.document_id,
                    doc_item_ref=a.doc_item_ref,
                    item_index=a.item_index,
                    page_no=a.page_no,
                    bbox=json.dumps(a.bbox) if a.bbox else None,
                    caption=a.caption,
                    description=a.description,
                    metadata=json.dumps(a.metadata or {}),
                    vector=a.embedding,
                )
            )
            a.id = asset_id

        table.add(records)
        return assets

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all multimodal assets for a document."""
        self.store._assert_writable()
        table = self._require_table()
        # Fast path: delete without pre-check; table.delete is idempotent.
        table.delete(f"document_id = '{document_id}'")
        return True

    async def search_by_vector(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filter: str | None = None,
    ) -> list[MMSearchResult]:
        """Vector search over multimodal assets."""
        table = self._require_table()

        q = table.search(query_vector).limit(limit)
        if filter:
            q = q.where(filter)

        # LanceDB returns _distance for vector search; smaller is better.
        # Convert to a similarity-like score for UX consistency.
        rows = q.to_list()
        results: list[MMSearchResult] = []
        for row in rows:
            asset_id = row.get("id")
            document_id = row.get("document_id")
            if not asset_id or not document_id:
                continue
            distance = float(row.get("_distance", 0.0))
            score = 1.0 / (1.0 + distance)
            bbox = None
            try:
                bbox = json.loads(row["bbox"]) if row.get("bbox") else None
            except Exception:
                bbox = None
            results.append(
                MMSearchResult(
                    asset_id=str(asset_id),
                    document_id=str(document_id),
                    score=score,
                    doc_item_ref=row.get("doc_item_ref", ""),
                    item_index=row.get("item_index"),
                    page_no=row.get("page_no"),
                    bbox=bbox,
                    caption=row.get("caption"),
                    description=row.get("description"),
                )
            )

        return results

