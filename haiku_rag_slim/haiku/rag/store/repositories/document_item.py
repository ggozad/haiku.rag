import json
from collections.abc import Mapping, Sequence

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

    async def resolve_refs_grouped(
        self, refs_by_document: "Mapping[str, Sequence[str]]"
    ) -> dict[str, dict[str, int]]:
        """Resolve self_refs to positions, across documents, in one query."""
        predicate = self._per_document_predicate(refs_by_document, "self_ref")
        if predicate is None:
            return {}
        rows = await (
            self.store.document_items_table.query()
            .select(["document_id", "self_ref", "position"])
            .where(predicate)
            .to_list()
        )
        grouped: dict[str, dict[str, int]] = {}
        for row in rows:
            grouped.setdefault(row["document_id"], {})[row["self_ref"]] = row[
                "position"
            ]
        return grouped

    async def get_items_in_ranges(
        self, ranges_by_document: "Mapping[str, tuple[int, int]]"
    ) -> dict[str, list[DocumentItem]]:
        """Items within a position range per document, in one query.

        Each document keeps its own inclusive range. Positions repeat across
        documents, so a shared range would splice one document's items into
        another's context.
        """
        clauses = []
        for document_id, (start, end) in ranges_by_document.items():
            safe_id = escape_sql_string(document_id)
            clauses.append(
                f"(document_id = '{safe_id}' "
                f"AND position >= {start} AND position <= {end})"
            )
        if not clauses:
            return {}
        rows = await (
            self.store.document_items_table.query()
            .select(_METADATA_COLUMNS)
            .where(" OR ".join(clauses))
            .to_list()
        )
        grouped: dict[str, list[DocumentItem]] = {}
        for row in rows:
            item = self._record_to_item(row)
            grouped.setdefault(item.document_id, []).append(item)
        for items in grouped.values():
            items.sort(key=lambda x: x.position)
        return grouped

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

    @staticmethod
    def _per_document_predicate(
        refs_by_document: "Mapping[str, Sequence[str | int]]", column: str
    ) -> str | None:
        """`(document_id = 'a' AND col IN (...)) OR (document_id = 'b' AND ...)`.

        Per-document rather than `col IN (union)`: self_ref and position values
        repeat across documents, so a union predicate would return other
        documents' rows, which for picture_data means fetching blobs nobody asked
        for. Returns None when nothing is asked for.
        """
        clauses = []
        for document_id, refs in refs_by_document.items():
            if not refs:
                continue
            safe_id = escape_sql_string(document_id)
            values = ", ".join(
                str(r) if isinstance(r, int) else f"'{escape_sql_string(r)}'"
                for r in refs
            )
            clauses.append(f"(document_id = '{safe_id}' AND {column} IN ({values}))")
        return " OR ".join(clauses) if clauses else None

    async def get_pictures_grouped(
        self, refs_by_document: "Mapping[str, list[str]]"
    ) -> tuple[dict[str, dict[str, bytes]], dict[str, dict[str, str]]]:
        """Picture bytes and their text, across documents, in one query.

        Returns `(bytes_by_document, text_by_document)`, each
        `{document_id: {self_ref: value}}` and each omitting refs whose value is
        empty. The text comes from the same rows as the bytes, so asking for it
        separately would be a second read of rows already in hand.
        """
        predicate = self._per_document_predicate(refs_by_document, "self_ref")
        if predicate is None:
            return {}, {}
        rows = await (
            self.store.document_items_table.query()
            .select(["document_id", "self_ref", "picture_data", "text"])
            .where(predicate)
            .to_list()
        )
        blobs: dict[str, dict[str, bytes]] = {}
        texts: dict[str, dict[str, str]] = {}
        for row in rows:
            data = row.get("picture_data")
            if not data:
                continue
            blobs.setdefault(row["document_id"], {})[row["self_ref"]] = data
            text = row.get("text")
            if text:
                texts.setdefault(row["document_id"], {})[row["self_ref"]] = text
        return blobs, texts

    async def get_caption_picture_refs_grouped(
        self, refs_by_document: "Mapping[str, list[str]]"
    ) -> dict[str, dict[str, str]]:
        """Map caption refs to the picture preceding them, in two queries.

        Two rather than one because the stages are dependent: a caption's
        picture is the item at `position - 1`, which the first query is what
        establishes.
        """
        predicate = self._per_document_predicate(refs_by_document, "self_ref")
        if predicate is None:
            return {}
        caption_rows = await (
            self.store.document_items_table.query()
            .select(["document_id", "self_ref", "position"])
            .where(f"label = 'caption' AND ({predicate})")
            .to_list()
        )
        if not caption_rows:
            return {}

        prev_to_caption: dict[str, dict[int, str]] = {}
        for row in caption_rows:
            prev_to_caption.setdefault(row["document_id"], {})[row["position"] - 1] = (
                row["self_ref"]
            )

        # Non-empty: every caption row contributed a position.
        picture_predicate = self._per_document_predicate(
            {did: list(positions) for did, positions in prev_to_caption.items()},
            "position",
        )
        picture_rows = await (
            self.store.document_items_table.query()
            .select(["document_id", "self_ref", "position"])
            .where(f"label = 'picture' AND ({picture_predicate})")
            .to_list()
        )
        grouped: dict[str, dict[str, str]] = {}
        for row in picture_rows:
            caption = prev_to_caption.get(row["document_id"], {}).get(row["position"])
            if caption:
                grouped.setdefault(row["document_id"], {})[caption] = row["self_ref"]
        return grouped
