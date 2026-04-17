def _build_document_filter(document_name: str) -> str:
    """Build SQL filter for document name matching.

    Matches against both uri and title fields, case-insensitive.
    Also matches without spaces to handle cases like "TB MED 593" vs "TBMED593".
    """
    escaped = document_name.replace("'", "''")
    no_spaces = escaped.replace(" ", "")
    return (
        f"LOWER(uri) LIKE LOWER('%{escaped}%') OR LOWER(title) LIKE LOWER('%{escaped}%') "
        f"OR LOWER(uri) LIKE LOWER('%{no_spaces}%') OR LOWER(title) LIKE LOWER('%{no_spaces}%')"
    )


def build_multi_document_filter(document_names: list[str]) -> str | None:
    """Build SQL filter for multiple document names (OR combined).

    Returns None if the list is empty.
    """
    if not document_names:
        return None
    filters = [_build_document_filter(name) for name in document_names]
    if len(filters) == 1:
        return filters[0]
    return " OR ".join(f"({f})" for f in filters)
