import json

from haiku.rag.store.engine import DocumentItemRecord, Store
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.utils import escape_sql_string


class DocumentItemRepository:
    """Repository for DocumentItem operations."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def _record_to_item(self, row: dict) -> DocumentItem:
        return DocumentItem(
            document_id=row["document_id"],
            position=row["position"],
            self_ref=row["self_ref"],
            label=row.get("label", ""),
            text=row.get("text", ""),
            page_numbers=json.loads(row.get("page_numbers", "[]")),
        )

    async def create_items(self, document_id: str, items: list[DocumentItem]) -> None:
        """Bulk insert items for a document."""
        if not items:
            return

        self.store._assert_writable()
        records = [
            DocumentItemRecord(
                document_id=document_id,
                position=item.position,
                self_ref=item.self_ref,
                label=item.label,
                text=item.text,
                page_numbers=json.dumps(item.page_numbers),
            )
            for item in items
        ]
        self.store.document_items_table.add(records)

    async def get_all_items(self, document_id: str) -> list[DocumentItem]:
        """Get all items for a document, sorted by position."""
        safe_id = escape_sql_string(document_id)
        rows = (
            self.store.document_items_table.search()
            .where(f"document_id = '{safe_id}'")
            .to_list()
        )
        items = [self._record_to_item(row) for row in rows]
        items.sort(key=lambda x: x.position)
        return items

    async def get_all_items_grouped(
        self, document_ids: list[str] | None = None
    ) -> dict[str, list[DocumentItem]]:
        """Get all items grouped by document_id in a single query.

        Args:
            document_ids: If provided, only fetch items for these documents.
                If None, fetches all items.

        Returns:
            Dict mapping document_id to sorted list of DocumentItem.
        """
        query = self.store.document_items_table.search()
        if document_ids is not None:
            safe_ids = ", ".join(f"'{escape_sql_string(did)}'" for did in document_ids)
            query = query.where(f"document_id IN ({safe_ids})")
        rows = query.to_list()

        grouped: dict[str, list[DocumentItem]] = {}
        for row in rows:
            item = self._record_to_item(row)
            grouped.setdefault(item.document_id, []).append(item)
        for items in grouped.values():
            items.sort(key=lambda x: x.position)
        return grouped

    async def get_items_in_range(
        self, document_id: str, start: int, end: int
    ) -> list[DocumentItem]:
        """Get items for a document within a position range (inclusive)."""
        safe_id = escape_sql_string(document_id)
        rows = (
            self.store.document_items_table.search()
            .where(
                f"document_id = '{safe_id}' "
                f"AND position >= {start} AND position <= {end}"
            )
            .to_list()
        )
        items = [self._record_to_item(row) for row in rows]
        items.sort(key=lambda x: x.position)
        return items

    async def resolve_refs(self, document_id: str, refs: list[str]) -> dict[str, int]:
        """Resolve self_refs to positions. Returns {self_ref: position}."""
        if not refs:
            return {}

        safe_id = escape_sql_string(document_id)
        refs_sql = ", ".join(f"'{escape_sql_string(r)}'" for r in refs)
        rows = (
            self.store.document_items_table.search()
            .select(["self_ref", "position"])
            .where(f"document_id = '{safe_id}' AND self_ref IN ({refs_sql})")
            .to_list()
        )
        return {row["self_ref"]: row["position"] for row in rows}

    async def get_item_count(self, document_id: str) -> int:
        """Count items for a document."""
        safe_id = escape_sql_string(document_id)
        return self.store.document_items_table.count_rows(
            filter=f"document_id = '{safe_id}'"
        )

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all items for a document."""
        self.store._assert_writable()
        safe_id = escape_sql_string(document_id)
        self.store.document_items_table.delete(f"document_id = '{safe_id}'")
