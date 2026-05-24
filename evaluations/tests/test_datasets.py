from pathlib import Path

from evaluations.datasets.mmlongbench import (
    build_mmlb_case,
    load_qa_records,
    map_mmlb_document,
    map_mmlb_retrieval,
)
from evaluations.datasets.open_rag_bench import (
    build_orb_case,
    download_pdf,
    is_multimodal_query,
    map_orb_document,
    map_orb_retrieval,
)
from evaluations.datasets.wix import (
    build_wix_case,
    map_wix_document,
    map_wix_retrieval,
)


class TestWix:
    def test_map_document_with_all_fields(self) -> None:
        doc = {
            "id": 123,
            "url": "https://wix.com/article",
            "html_content": "<p>Content</p>",
            "title": "My Article",
        }
        payload = map_wix_document(doc)
        assert payload.uri == "123"
        assert payload.content == "<p>Content</p>"
        assert payload.title == "My Article"
        assert payload.format == "html"
        assert payload.metadata == {
            "article_id": "123",
            "url": "https://wix.com/article",
        }

    def test_map_document_no_id(self) -> None:
        doc = {
            "id": None,
            "url": "https://wix.com/page",
            "html_content": "<p>Text</p>",
            "title": None,
        }
        payload = map_wix_document(doc)
        assert payload.uri == "https://wix.com/page"

    def test_map_document_no_metadata(self) -> None:
        doc = {"id": None, "url": None, "html_content": "<p>X</p>", "title": None}
        payload = map_wix_document(doc)
        assert payload.metadata is None

    def test_map_retrieval(self) -> None:
        doc = {"question": "How to add a page?", "article_ids": [10, 20]}
        sample = map_wix_retrieval(doc)
        assert sample is not None
        assert sample.question == "How to add a page?"
        assert sample.expected_uris == ("10", "20")

    def test_map_retrieval_no_article_ids(self) -> None:
        doc = {"question": "Q?", "article_ids": None}
        assert map_wix_retrieval(doc) is None

    def test_map_retrieval_empty_article_ids(self) -> None:
        doc = {"question": "Q?", "article_ids": []}
        assert map_wix_retrieval(doc) is None

    def test_build_case(self) -> None:
        doc = {
            "question": "How?",
            "answer": "Like this.",
            "article_ids": [5, 10],
        }
        case = build_wix_case(2, doc)
        assert case.name == "2_5-10"
        assert case.inputs == "How?"
        assert case.expected_output == "Like this."
        assert case.metadata is not None
        assert case.metadata["case_index"] == "2"

    def test_build_case_no_article_ids(self) -> None:
        doc = {"question": "Q?", "answer": "A.", "article_ids": None}
        case = build_wix_case(1, doc)
        assert case.name == "case_1"


class TestOpenRAGBench:
    def test_map_document(self, tmp_path: Path) -> None:
        # Pre-create a cached PDF
        cache_dir = tmp_path / "pdfs"
        cache_dir.mkdir()
        pdf_path = cache_dir / "paper1.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        doc = {"paper_id": "paper1", "pdf_url": "https://example.com/paper1.pdf"}
        # Patch get_cache_dir to use our tmp_path
        from unittest.mock import patch

        with patch(
            "evaluations.datasets.open_rag_bench.get_cache_dir", return_value=cache_dir
        ):
            payload = map_orb_document(doc)

        assert payload is not None
        assert payload.uri == "paper1"
        assert payload.title == "paper1"
        assert payload.source_path == pdf_path
        assert payload.metadata == {"arxiv_id": "paper1"}

    def test_map_document_download_fails(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "pdfs"
        cache_dir.mkdir()

        doc = {"paper_id": "missing", "pdf_url": "https://example.com/missing.pdf"}
        from unittest.mock import patch

        with patch(
            "evaluations.datasets.open_rag_bench.get_cache_dir", return_value=cache_dir
        ):
            with patch(
                "evaluations.datasets.open_rag_bench.download_pdf", return_value=None
            ):
                payload = map_orb_document(doc)

        assert payload is None

    def test_map_retrieval(self) -> None:
        doc = {
            "query": "What is attention?",
            "doc_id": "1706.03762",
            "source": "text",
        }
        sample = map_orb_retrieval(doc)
        assert sample is not None
        assert sample.question == "What is attention?"
        assert sample.expected_uris == ("1706.03762",)
        assert sample.source_type == "text"

    def test_build_case(self) -> None:
        doc = {
            "query_id": "q_abcdef12",
            "query": "Explain transformers.",
            "answer": "Transformers are...",
            "type": "factual",
            "source": "text",
        }
        case = build_orb_case(1, doc)
        assert case.name == "1_q_abcdef"
        assert case.inputs == "Explain transformers."
        assert case.expected_output == "Transformers are..."
        assert case.metadata is not None
        assert case.metadata["query_id"] == "q_abcdef12"

    def test_download_pdf_uses_cache(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "cached.pdf"
        pdf_path.write_bytes(b"%PDF-cached")
        result = download_pdf("cached", "https://example.com/cached.pdf", tmp_path)
        assert result == pdf_path

    def test_is_multimodal_query(self) -> None:
        assert is_multimodal_query("image") is True
        assert is_multimodal_query("image_table") is True
        assert is_multimodal_query("text") is False


class TestMMLongBenchDoc:
    def test_map_document(self, tmp_path: Path) -> None:
        pdf_dir = tmp_path / "documents"
        pdf_dir.mkdir()
        pdf_path = pdf_dir / "report.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        from unittest.mock import patch

        with patch(
            "evaluations.datasets.mmlongbench.get_cache_dir",
            return_value=tmp_path,
        ):
            payload = map_mmlb_document(
                {"doc_id": "report.pdf", "doc_type": "Financial report"}
            )

        assert payload is not None
        assert payload.uri == "report.pdf"
        assert payload.title == "report.pdf"
        assert payload.source_path == pdf_path
        assert payload.metadata == {"doc_type": "Financial report"}

    def test_map_document_missing_pdf(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        with patch(
            "evaluations.datasets.mmlongbench.get_cache_dir",
            return_value=tmp_path,
        ):
            payload = map_mmlb_document(
                {"doc_id": "missing.pdf", "doc_type": "Brochure"}
            )

        assert payload is None

    def test_map_retrieval(self) -> None:
        doc = {
            "question": "What is the revenue?",
            "doc_id": "NIKE_2021_10K.pdf",
            "evidence_pages": [3, 5],
            "evidence_sources": ["Table", "Pure-text"],
        }
        sample = map_mmlb_retrieval(doc)
        assert sample is not None
        assert sample.question == "What is the revenue?"
        assert sample.expected_uris == ("NIKE_2021_10K.pdf",)
        assert sample.source_type == "Table,Pure-text"

    def test_map_retrieval_skips_unanswerable(self) -> None:
        doc = {
            "question": "What does the document say about Mars?",
            "doc_id": "NIKE_2021_10K.pdf",
            "evidence_pages": [],
            "evidence_sources": [],
        }
        assert map_mmlb_retrieval(doc) is None

    def test_build_case(self) -> None:
        doc = {
            "doc_id": "report.pdf",
            "doc_type": "Financial report",
            "question": "What is the net income?",
            "answer": "42",
            "evidence_pages": [5],
            "evidence_sources": ["Table"],
            "answer_format": "Int",
        }
        case = build_mmlb_case(7, doc)
        assert case.name == "7_report.pdf"
        assert case.inputs == "What is the net income?"
        assert case.expected_output == "42"
        assert case.metadata is not None
        assert case.metadata["doc_id"] == "report.pdf"
        assert case.metadata["doc_type"] == "Financial report"
        assert case.metadata["answer_format"] == "Int"
        assert case.metadata["evidence_pages"] == "[5]"
        assert case.metadata["evidence_sources"] == "Table"
        assert case.metadata["case_index"] == "7"

    def test_load_qa_records_parses_list_fields(self) -> None:
        from unittest.mock import patch

        raw_rows = [
            {
                "doc_id": "a.pdf",
                "doc_type": "Brochure",
                "question": "Q1?",
                "answer": "A1",
                "evidence_pages": "[3, 5]",
                "evidence_sources": "['Table', 'Pure-text']",
                "answer_format": "Str",
            },
            {
                "doc_id": "b.pdf",
                "doc_type": "Academic paper",
                "question": "Q2?",
                "answer": "Not answerable",
                "evidence_pages": "[]",
                "evidence_sources": "[]",
                "answer_format": "None",
            },
        ]

        with patch(
            "evaluations.datasets.mmlongbench._load_hf_qa_split",
            return_value=raw_rows,
        ):
            records = load_qa_records()

        assert records[0]["evidence_pages"] == [3, 5]
        assert records[0]["evidence_sources"] == ["Table", "Pure-text"]
        assert records[1]["evidence_pages"] == []
        assert records[1]["evidence_sources"] == []
