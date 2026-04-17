from haiku.rag.tools.filters import _build_document_filter, build_multi_document_filter


def test_build_document_filter_simple():
    """Test _build_document_filter with simple name."""
    result = _build_document_filter("mytest")
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result


def test_build_document_filter_with_spaces():
    """Test _build_document_filter handles spaces correctly."""
    result = _build_document_filter("TB MED 593")
    assert "LOWER(uri) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(uri) LIKE LOWER('%TBMED593%')" in result
    assert "LOWER(title) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(title) LIKE LOWER('%TBMED593%')" in result


def test_build_document_filter_escapes_quotes():
    """Test _build_document_filter escapes single quotes."""
    result = _build_document_filter("O'Reilly")
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
    assert " OR (" not in result


def test_build_multi_document_filter_multiple():
    """Test build_multi_document_filter with multiple documents."""
    result = build_multi_document_filter(["doc1", "doc2"])
    assert result is not None
    assert "doc1" in result
    assert "doc2" in result
    assert " OR (" in result
