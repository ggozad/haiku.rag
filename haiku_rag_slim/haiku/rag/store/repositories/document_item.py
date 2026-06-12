import json

from haiku.rag.store.engine import DocumentItemRecord, Store
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.utils import escape_sql_string

# Per-item metadata columns. The payload column ``picture_data`` is fetched
# explicitly via ``get_picture_bytes`` / ``get_pictures_for_chunk`` /
# ``get_all_picture_data`` so bulk scans don't pull MB-scale image bytes.
_METADATA_COLUMNS = [
    "document_id",
    "position",
    "self_ref",
    "label",
    "text",
    "page_numbers",
    "heading_level",
    "tree_depth",
]


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
            heading_level=row.get("heading_level", 0) or 0,
            tree_depth=row.get("tree_depth", 0) or 0,
        )

    def _to_record(self, document_id: str, item: DocumentItem) -> DocumentItemRecord:
        return DocumentItemRecord(
            document_id=document_id,
            position=item.position,
            self_ref=item.self_ref,
            label=item.label,
            text=item.text,
            page_numbers=json.dumps(item.page_numbers),
            picture_data=item.picture_data,
            heading_level=item.heading_level,
            tree_depth=item.tree_depth,
        )

    async def create_items(self, document_id: str, items: list[DocumentItem]) -> None:
        """Bulk insert items for a document."""
        if not items:
            return

        self.store._assert_writable()
        records = [self._to_record(document_id, item) for item in items]
        await self.store.document_items_table.add(records)

    async def create_all(self, items: list[DocumentItem]) -> None:
        """Bulk insert items spanning any number of documents in a single
        table version, keyed by each item's own ``document_id``."""
        if not items:
            return

        self.store._assert_writable()
        records = [self._to_record(item.document_id, item) for item in items]
        await self.store.document_items_table.add(records)

    async def replace_for_document(
        self, document_id: str, items: list[DocumentItem]
    ) -> None:
        """Replace all items for a document with one scoped merge operation."""
        self.store._assert_writable()

        if not items:
            await self.delete_by_document_id(document_id)
            return

        for item in items:
            assert item.document_id == document_id, (
                "All items must belong to the replaced document"
            )

        safe_id = escape_sql_string(document_id)
        records = [self._to_record(document_id, item) for item in items]
        await (
            self.store.document_items_table.merge_insert(["document_id", "self_ref"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete(f"document_id = '{safe_id}'")
            .execute(records)
        )

    async def get_all_items(self, document_id: str) -> list[DocumentItem]:
        """Get all items for a document, sorted by position."""
        safe_id = escape_sql_string(document_id)
        rows = await (
            self.store.document_items_table.query()
            .select(_METADATA_COLUMNS)
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
        query = self.store.document_items_table.query().select(_METADATA_COLUMNS)
        if document_ids is not None:
            safe_ids = ", ".join(f"'{escape_sql_string(did)}'" for did in document_ids)
            query = query.where(f"document_id IN ({safe_ids})")
        rows = await query.to_list()

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
        rows = await (
            self.store.document_items_table.query()
            .select(_METADATA_COLUMNS)
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
        rows = await (
            self.store.document_items_table.query()
            .select(["self_ref", "position"])
            .where(f"document_id = '{safe_id}' AND self_ref IN ({refs_sql})")
            .to_list()
        )
        return {row["self_ref"]: row["position"] for row in rows}

    async def get_item_count(self, document_id: str) -> int:
        """Count items for a document."""
        safe_id = escape_sql_string(document_id)
        return await self.store.document_items_table.count_rows(
            filter=f"document_id = '{safe_id}'"
        )

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all items for a document."""
        self.store._assert_writable()
        safe_id = escape_sql_string(document_id)
        await self.store.document_items_table.delete(f"document_id = '{safe_id}'")

    async def get_picture_bytes(self, document_id: str, self_ref: str) -> bytes | None:
        """Fetch raw picture bytes for a single picture item by self_ref."""
        safe_id = escape_sql_string(document_id)
        safe_ref = escape_sql_string(self_ref)
        rows = await (
            self.store.document_items_table.query()
            .select(["picture_data"])
            .where(f"document_id = '{safe_id}' AND self_ref = '{safe_ref}'")
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        return rows[0].get("picture_data")

    async def get_all_picture_data(self, document_id: str) -> dict[str, bytes]:
        """Snapshot every picture row's bytes for a single document.

        Returns ``{self_ref: picture_data}`` for every row whose
        ``picture_data`` is non-null. Used by rebuild / update flows to
        preserve picture bytes across a delete-and-re-extract cycle when the
        live docling document has already been stripped of its picture URIs.
        """
        safe_id = escape_sql_string(document_id)
        rows = await (
            self.store.document_items_table.query()
            .select(["self_ref", "picture_data"])
            .where(f"document_id = '{safe_id}'")
            .to_list()
        )
        result: dict[str, bytes] = {}
        for row in rows:
            data = row.get("picture_data")
            if data:
                result[row["self_ref"]] = data
        return result

    async def get_pictures_for_chunk(
        self, document_id: str, refs: list[str]
    ) -> dict[str, bytes]:
        """Fetch picture bytes for multiple self_refs within a single document.

        Returns a mapping of self_ref → bytes, including only refs that have
        non-null picture_data. Refs without bytes (or unknown refs) are omitted.
        """
        if not refs:
            return {}

        safe_id = escape_sql_string(document_id)
        refs_sql = ", ".join(f"'{escape_sql_string(r)}'" for r in refs)
        rows = await (
            self.store.document_items_table.query()
            .select(["self_ref", "picture_data"])
            .where(f"document_id = '{safe_id}' AND self_ref IN ({refs_sql})")
            .to_list()
        )
        result: dict[str, bytes] = {}
        for row in rows:
            data = row.get("picture_data")
            if data:
                result[row["self_ref"]] = data
        return result

    async def get_text_for_refs(
        self, document_id: str, refs: list[str]
    ) -> dict[str, str]:
        """Fetch the ``text`` field for multiple self_refs within a single document.

        Returns ``{self_ref: text}`` for refs whose text is non-empty. Used
        alongside ``get_pictures_for_chunk`` to label figures in agent-facing
        search results: picture items carry their VLM-generated caption in
        the ``text`` field, and the OpenAI vision message format has no
        identifier on binary parts, so the caption text is the only signal a
        model can use to correlate a description with the picture it sees.
        """
        if not refs:
            return {}

        safe_id = escape_sql_string(document_id)
        refs_sql = ", ".join(f"'{escape_sql_string(r)}'" for r in refs)
        rows = await (
            self.store.document_items_table.query()
            .select(["self_ref", "text"])
            .where(f"document_id = '{safe_id}' AND self_ref IN ({refs_sql})")
            .to_list()
        )
        result: dict[str, str] = {}
        for row in rows:
            text = row.get("text") or ""
            if text:
                result[row["self_ref"]] = text
        return result
