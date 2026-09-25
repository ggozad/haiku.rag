def escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL string literals."""
    return value.replace("'", "''")


def build_document_id_filter(document_ids: list[str]) -> str | None:
    """SQL filter matching exactly these documents, or None for an empty list."""
    if not document_ids:
        return None
    ids = ", ".join(f"'{escape_sql_string(i)}'" for i in document_ids)
    return f"id IN ({ids})"
