def escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL string literals."""
    return value.replace("'", "''")


def escape_like_pattern(value: str) -> str:
    """Escape `LIKE` wildcards so `value` matches literally under `ESCAPE '\\'`."""
    for character in ("\\", "%", "_"):
        value = value.replace(character, f"\\{character}")
    return value


def build_document_id_filter(document_ids: list[str]) -> str | None:
    """SQL filter matching exactly these documents, or None for an empty list."""
    if not document_ids:
        return None
    ids = ", ".join(f"'{escape_sql_string(i)}'" for i in document_ids)
    return f"id IN ({ids})"


def title_or_uri_filter(term: str) -> str | None:
    """A document filter matching `term` in a title or URI, or None for no term.

    The term is text to find, not a pattern: `LIKE` is case-sensitive and reads
    `%` and `_` as wildcards, so both sides are lowered and the term's own
    wildcards are escaped.
    """
    term = term.strip()
    if not term:
        return None
    escaped = escape_sql_string(f"%{escape_like_pattern(term.lower())}%")
    return (
        f"LOWER(title) LIKE '{escaped}' ESCAPE '\\' "
        f"OR LOWER(uri) LIKE '{escaped}' ESCAPE '\\'"
    )
