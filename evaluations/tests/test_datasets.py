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
from evaluations.datasets.repliqa import (
    build_repliqa_case,
    map_repliqa_document,
    map_repliqa_retrieval,
)
from evaluations.datasets.wix import (
    build_wix_case,
    map_wix_document,
    map_wix_retrieval,
)


class TestRepliqa:
    def test_map_document(self) -> None:
        doc = {"document_id": "doc-42", "document_extracted": "Some content here."}
        payload = map_repliqa_document(doc)
        assert payload.uri == "doc-42"
        assert payload.content == "Some content here."

    def test_map_retrieval(self) -> None:
        doc = {
            "question": "What happened?",
            "answer": "Something happened.",
            "document_id": "doc-42",
        }
        sample = map_repliqa_retrieval(doc)
        assert sample is not None
        assert sample.question == "What happened?"
        assert sample.expected_uris == ("doc-42",)

    def test_map_retrieval_skips_unanswerable(self) -> None:
        doc = {
            "question": "What?",
            "answer": "The answer is not found in the document.",
            "document_id": "doc-1",
        }
        assert map_repliqa_retrieval(doc) is None

    def test_build_case(self) -> None:
        doc = {
            "document_id": "doc-7",
            "question": "Why?",
            "answer": "Because.",
        }
        case = build_repliqa_case(3, doc)
        assert case.name == "3_doc-7"
        assert case.inputs == "Why?"
        assert case.expected_output == "Because."
        assert case.metadata == {"document_id": "doc-7", "case_index": "3"}

    def test_build_case_none_document_id(self) -> None:
        doc = {"document_id": None, "question": "Q?", "answer": "A."}
        case = build_repliqa_case(1, doc)
        assert case.name == "case_1"


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
        docs = extract_unique_documents(dataset)  # type: ignore[arg-type]
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
