from haiku.rag.tools.filters import (
    build_document_filter,
    build_multi_document_filter,
    combine_filters,
)


def test_build_document_filter_simple():
    """Test build_document_filter with simple name."""
    result = build_document_filter("mytest")
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result


def test_build_document_filter_with_spaces():
    """Test build_document_filter handles spaces correctly."""
    result = build_document_filter("TB MED 593")
    # Should include both the original (with spaces) and without spaces
    assert "LOWER(uri) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(uri) LIKE LOWER('%TBMED593%')" in result
    assert "LOWER(title) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(title) LIKE LOWER('%TBMED593%')" in result


def test_build_document_filter_escapes_quotes():
    """Test build_document_filter escapes single quotes."""
    result = build_document_filter("O'Reilly")
    # Single quotes should be doubled for SQL escaping
    assert "O''Reilly" in result


def test_build_multi_document_filter_empty():
    """Test build_multi_document_filter returns None for empty list."""
    result = build_multi_document_filter([])
    assert result is None


def test_build_multi_document_filter_single():
    """Test build_multi_document_filter with single document."""
    result = build_multi_document_filter(["mytest"])
    assert result is not None
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result
    # Single document should not have extra wrapping parentheses
    assert " OR (" not in result


def test_build_multi_document_filter_multiple():
    """Test build_multi_document_filter with multiple documents."""
    result = build_multi_document_filter(["doc1", "doc2"])
    assert result is not None
    # Should have OR-combined filters
    assert "doc1" in result
    assert "doc2" in result
    assert " OR (" in result


def test_combine_filters_both_none():
    """Test combine_filters with both None."""
    result = combine_filters(None, None)
    assert result is None


def test_combine_filters_first_only():
    """Test combine_filters with only first filter."""
    result = combine_filters("uri = 'test'", None)
    assert result == "uri = 'test'"


def test_combine_filters_second_only():
    """Test combine_filters with only second filter."""
    result = combine_filters(None, "title = 'doc'")
    assert result == "title = 'doc'"


def test_combine_filters_both():
    """Test combine_filters combines with AND."""
    result = combine_filters("uri = 'test'", "title = 'doc'")
    assert result == "(uri = 'test') AND (title = 'doc')"
