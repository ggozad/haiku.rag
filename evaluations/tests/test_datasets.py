from pathlib import Path

from evaluations.datasets.hotpotqa import (
    build_hotpotqa_case,
    extract_unique_documents,
    map_hotpotqa_document,
    map_hotpotqa_retrieval,
)
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


class TestHotpotQA:
    def test_map_document(self) -> None:
        doc = {"title": "Albert Einstein", "content": "Was a physicist."}
        payload = map_hotpotqa_document(doc)
        assert payload.uri == "Albert Einstein"
        assert payload.content == "Was a physicist."
        assert payload.title == "Albert Einstein"

    def test_map_retrieval(self) -> None:
        doc = {
            "question": "Who was Einstein?",
            "supporting_facts": {"title": ["Albert Einstein", "Physics"]},
        }
        sample = map_hotpotqa_retrieval(doc)
        assert sample is not None
        assert sample.expected_uris == ("Albert Einstein", "Physics")

    def test_map_retrieval_deduplicates_titles(self) -> None:
        doc = {
            "question": "Q?",
            "supporting_facts": {"title": ["A", "B", "A"]},
        }
        sample = map_hotpotqa_retrieval(doc)
        assert sample is not None
        assert sample.expected_uris == ("A", "B")

    def test_map_retrieval_no_titles(self) -> None:
        doc = {"question": "Q?", "supporting_facts": {"title": []}}
        assert map_hotpotqa_retrieval(doc) is None

    def test_build_case(self) -> None:
        doc = {
            "id": "abc123",
            "question": "What is X?",
            "answer": "X is Y.",
            "type": "comparison",
            "level": "hard",
        }
        case = build_hotpotqa_case(5, doc)
        assert case.name == "5_abc123"
        assert case.inputs == "What is X?"
        assert case.expected_output == "X is Y."
        assert case.metadata == {
            "question_id": "abc123",
            "type": "comparison",
            "level": "hard",
            "case_index": "5",
        }

    def test_extract_unique_documents(self) -> None:
        # Simulate a minimal dataset with context
        dataset = [
            {
                "context": {
                    "title": ["Doc A", "Doc B"],
                    "sentences": [["Sentence 1."], ["Sentence 2.", " More."]],
                }
            },
            {
                "context": {
                    "title": ["Doc A", "Doc C"],
                    "sentences": [["Dupe."], ["Sentence 3."]],
                }
            },
        ]
        docs = extract_unique_documents(dataset)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert len(docs) == 3
        titles = [d["title"] for d in docs]
        assert titles == ["Doc A", "Doc B", "Doc C"]
        assert docs[1]["content"] == "Sentence 2.  More."


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

    def test_pdf_repo_path_per_subset(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        blob = tmp_path / "blob"
        blob.write_bytes(b"%PDF-fake")
        cache = tmp_path / "cache"
        cache.mkdir()
        cases = [
            (
                "FinQA",
                "dev",
                "pdf/V/2008/page_17.pdf",
                "data/FinQA/dev/pdf/V/2008/page_17.pdf",
            ),
            ("TAT-DQA", "dev", "raw/abc123.pdf", "data/TAT-DQA/dev/raw/abc123.pdf"),
        ]
        for subset, split, file_name, expected_repo_path in cases:
            with (
                patch(
                    "evaluations.datasets.t2_ragbench.get_cache_dir",
                    return_value=cache,
                ),
                patch(
                    "evaluations.datasets.t2_ragbench.hf_hub_download",
                    return_value=str(blob),
                ) as dl,
            ):
                download_t2_pdf(subset, split, file_name)
            assert dl.call_args.args[1] == expected_repo_path

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
