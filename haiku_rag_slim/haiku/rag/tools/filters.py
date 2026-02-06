from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haiku.rag.tools.context import ToolContext


def build_document_filter(document_name: str) -> str:
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
    filters = [build_document_filter(name) for name in document_names]
    if len(filters) == 1:
        return filters[0]
    return " OR ".join(f"({f})" for f in filters)


def get_session_filter(
    context: "ToolContext | None",
    base_filter: str | None = None,
) -> str | None:
    """Build effective filter from session state document filter and base filter.

    Checks the ToolContext for a registered SessionState. If it has a
    document_filter, builds a SQL filter from it and combines with base_filter.

    Args:
        context: Optional ToolContext that may contain a SessionState.
        base_filter: Optional base SQL WHERE clause to combine with.

    Returns:
        Combined filter string, or None if no filters apply.
    """
    if context is None:
        return base_filter

    from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState

    session_state = context.get_typed(SESSION_NAMESPACE, SessionState)
    if session_state is None or not session_state.document_filter:
        return base_filter

    session_filter = build_multi_document_filter(session_state.document_filter)
    return combine_filters(base_filter, session_filter)


def combine_filters(filter1: str | None, filter2: str | None) -> str | None:
    """Combine two SQL filters with AND logic.

    Returns None if both filters are None.
    """
    filters = [f for f in [filter1, filter2] if f]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return f"({filters[0]}) AND ({filters[1]})"
