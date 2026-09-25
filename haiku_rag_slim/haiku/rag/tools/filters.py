from haiku.rag.utils import escape_sql_string


def build_document_id_filter(document_ids: list[str]) -> str | None:
    """SQL filter matching exactly these documents, or None for an empty list.

    Ids repeat only between copies of a database, where the same id names the
    same document in each.
    """
    if not document_ids:
        return None
    ids = ", ".join(f"'{escape_sql_string(i)}'" for i in document_ids)
    return f"id IN ({ids})"
