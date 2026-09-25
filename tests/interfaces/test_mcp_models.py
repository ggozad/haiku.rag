from haiku.rag.mcp import DocumentInfo


class TestDocumentModels:
    """Tests for document models."""

    def test_document_info(self):
        """DocumentInfo holds basic document metadata."""
        info = DocumentInfo(title="Test Doc", uri="test://doc", created="2024-01-01")
        assert info.title == "Test Doc"
        assert info.uri == "test://doc"
        assert info.created == "2024-01-01"
