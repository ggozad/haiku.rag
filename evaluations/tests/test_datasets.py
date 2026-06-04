from pathlib import Path

from evaluations.datasets.open_rag_bench import (
    build_orb_case,
    download_pdf,
    is_multimodal_query,
    map_orb_document,
    map_orb_retrieval,
)
from evaluations.datasets.t2_ragbench import (
    build_t2_case,
    download_t2_pdf,
    load_t2_corpus,
    map_t2_document,
    map_t2_retrieval,
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


class TestT2RAGBench:
    def _row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "id": "finqa_dev_0",
            "context_id": "finqa_dev_ctx_138",
            "subset": "FinQA",
            "split": "dev",
            "file_name": "pdf/V/2008/page_17.pdf",
            "question": "What was the average payment volume per transaction?",
            "program_answer": "127.4",
            "company_name": "Visa Inc.",
            "company_symbol": "V",
            "report_year": 2008,
            "company_sector": "Financials",
        }
        row.update(overrides)
        return row

    def test_map_document(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "page_17.pdf"
        pdf_path.write_bytes(b"%PDF-fake")
        from unittest.mock import patch

        with patch(
            "evaluations.datasets.t2_ragbench.download_t2_pdf",
            return_value=pdf_path,
        ) as download:
            payload = map_t2_document(self._row())

        download.assert_called_once_with("FinQA", "dev", "pdf/V/2008/page_17.pdf")
        assert payload is not None
        assert payload.uri == "finqa_dev_ctx_138"
        assert payload.source_path == pdf_path
        assert payload.content is None
        assert payload.title == "Visa Inc. 2008"
        assert payload.metadata == {
            "file_name": "pdf/V/2008/page_17.pdf",
            "company_name": "Visa Inc.",
            "company_symbol": "V",
            "report_year": 2008,
            "company_sector": "Financials",
        }

    def test_map_document_title_falls_back_to_context_id(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "page.pdf"
        pdf_path.write_bytes(b"%PDF-fake")
        from unittest.mock import patch

        row = self._row(company_name=None, report_year=None)
        with patch(
            "evaluations.datasets.t2_ragbench.download_t2_pdf",
            return_value=pdf_path,
        ):
            payload = map_t2_document(row)

        assert payload is not None
        assert payload.title == "finqa_dev_ctx_138"
        assert payload.metadata is not None
        assert "company_name" not in payload.metadata
        assert "report_year" not in payload.metadata

    def test_map_retrieval(self) -> None:
        sample = map_t2_retrieval(self._row())
        assert sample is not None
        assert sample.question == self._row()["question"]
        assert sample.expected_uris == ("finqa_dev_ctx_138",)

    def test_build_case(self) -> None:
        case = build_t2_case(3, self._row())
        assert case.name == "3_finqa_dev_0"
        assert case.inputs == self._row()["question"]
        assert case.expected_output == "127.4"
        assert case.metadata is not None
        assert case.metadata["case_index"] == "3"
        assert case.metadata["context_id"] == "finqa_dev_ctx_138"

    def test_build_case_casts_numeric_answer(self) -> None:
        case = build_t2_case(0, self._row(program_answer=127.4))
        assert case.expected_output == "127.4"

    def test_download_pdf_materializes_pdf_suffix(self, tmp_path: Path) -> None:
        # hf_hub_download can return a suffix-less blob path; the cached copy
        # must carry the .pdf extension the converter dispatches on.
        blob = tmp_path / "blobs" / "6aa49306deadbeef"
        blob.parent.mkdir()
        blob.write_bytes(b"%PDF-fake")
        cache = tmp_path / "cache"
        cache.mkdir()
        from unittest.mock import patch

        with (
            patch("evaluations.datasets.t2_ragbench.get_cache_dir", return_value=cache),
            patch(
                "evaluations.datasets.t2_ragbench.hf_hub_download",
                return_value=str(blob),
            ),
        ):
            out = download_t2_pdf("FinQA", "dev", "pdf/V/2008/page_17.pdf")

        assert out.suffix == ".pdf"
        assert out.exists()
        assert out.read_bytes() == b"%PDF-fake"
        assert out.name == "FinQA_dev_pdf_V_2008_page_17.pdf"

    def test_load_corpus_dedupes_by_context_id(self) -> None:
        from unittest.mock import patch

        rows = [
            self._row(id="finqa_dev_0", context_id="ctx_a"),
            self._row(id="finqa_dev_1", context_id="ctx_a"),
            self._row(id="finqa_dev_2", context_id="ctx_b"),
        ]
        with patch("evaluations.datasets.t2_ragbench._load_rows", return_value=rows):
            corpus = load_t2_corpus("FinQA")

        assert len(corpus) == 2
        assert {r["context_id"] for r in corpus} == {"ctx_a", "ctx_b"}
