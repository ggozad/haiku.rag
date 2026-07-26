from pathlib import Path

import pytest

from evaluations.datasets.frames import (
    FETCH_ATTEMPTS,
    build_frames_case,
    fetch_article,
    map_frames_document,
    map_frames_retrieval,
    normalize_wiki_url,
    parse_revid,
    parse_wiki_links,
    question_is_answerable,
    strip_navigation,
)
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


class TestFrames:
    def test_parse_wiki_links_plain(self) -> None:
        raw = (
            "['https://en.wikipedia.org/wiki/James_Buchanan', "
            "'https://en.wikipedia.org/wiki/Harriet_Lane']"
        )
        assert parse_wiki_links(raw) == [
            "https://en.wikipedia.org/wiki/James_Buchanan",
            "https://en.wikipedia.org/wiki/Harriet_Lane",
        ]

    def test_parse_wiki_links_splits_comma_joined_urls(self) -> None:
        raw = (
            "['https://en.wikipedia.org/wiki/Tim_Salmon, "
            "https://en.wikipedia.org/wiki/Troy_Glaus, ']"
        )
        assert parse_wiki_links(raw) == [
            "https://en.wikipedia.org/wiki/Tim_Salmon",
            "https://en.wikipedia.org/wiki/Troy_Glaus",
        ]

    def test_parse_wiki_links_keeps_commas_inside_titles(self) -> None:
        raw = (
            "['https://en.wikipedia.org/wiki/Lincoln,_Nebraska', "
            "'https://en.wikipedia.org/wiki/Key_West#:~:text=The%20southernmost,"
            "apart%20at%20their%20closest%20points.']"
        )
        assert parse_wiki_links(raw) == [
            "https://en.wikipedia.org/wiki/Lincoln,_Nebraska",
            "https://en.wikipedia.org/wiki/Key_West#:~:text=The%20southernmost,"
            "apart%20at%20their%20closest%20points.",
        ]

    def test_parse_wiki_links_strips_trailing_annotation(self) -> None:
        raw = "['https://en.wikipedia.org/wiki/Pok%C3%A9mon (NOT REQUIRED, BUT HELPFUL) ']"
        assert parse_wiki_links(raw) == ["https://en.wikipedia.org/wiki/Pok%C3%A9mon"]

    def test_normalize_strips_fragment_and_mobile_host(self) -> None:
        assert (
            normalize_wiki_url("https://en.m.wikipedia.org/wiki/World_War_I#Aftermath")
            == "https://en.wikipedia.org/wiki/World_War_I"
        )

    def test_normalize_decodes_and_canonicalizes_title(self) -> None:
        assert (
            normalize_wiki_url("https://en.wikipedia.org/wiki/pain %26 Gain")
            == "https://en.wikipedia.org/wiki/Pain_&_Gain"
        )

    def test_normalize_schemeless(self) -> None:
        assert (
            normalize_wiki_url("en.wikipedia.org/wiki/Grazia_Deledda")
            == "https://en.wikipedia.org/wiki/Grazia_Deledda"
        )

    def test_normalize_index_php_title(self) -> None:
        assert (
            normalize_wiki_url(
                "https://en.wikipedia.org/w/index.php?title=Bronco&redirect=no"
            )
            == "https://en.wikipedia.org/wiki/Bronco"
        )

    def test_normalize_search_url(self) -> None:
        url = (
            "https://en.wikipedia.org/w/index.php?search=Polytrichum+piliferum"
            "&title=Special:Search&profile=advanced&fulltext=1&ns0=1"
        )
        assert (
            normalize_wiki_url(url)
            == "https://en.wikipedia.org/wiki/Polytrichum_piliferum"
        )

    def test_normalize_shortlink_passthrough(self) -> None:
        assert normalize_wiki_url("https://w.wiki/ASFv") == "https://w.wiki/ASFv"

    def test_normalize_rejects_non_article(self) -> None:
        assert normalize_wiki_url("") is None
        assert normalize_wiki_url("https://en.wikipedia.org/foo") is None

    def test_parse_revid(self) -> None:
        assert parse_revid('W/"1364811104/52cd04f4-864c-11f1"') == "1364811104"
        assert parse_revid('"1234/abc"') == "1234"
        assert parse_revid(None) is None
        assert parse_revid("") is None

    def test_strip_navigation_removes_navboxes_keeps_infobox(self) -> None:
        html = (
            "<html><body>"
            '<table class="infobox"><tbody><tr><td>Born April 23, 1791</td></tr></tbody></table>'
            "<p>Some prose.</p>"
            '<div role="navigation"><table><tbody><tr><td>v t e Presidents</td></tr></tbody></table></div>'
            "</body></html>"
        )
        stripped = strip_navigation(html)
        assert "Born April 23, 1791" in stripped
        assert "Some prose." in stripped
        assert "v t e Presidents" not in stripped

    def test_map_retrieval_normalizes_and_dedupes(self) -> None:
        row = {
            "Prompt": "Who was the 15th president?",
            "wiki_links": (
                "['https://en.wikipedia.org/wiki/James_Buchanan#Presidency', "
                "'https://en.m.wikipedia.org/wiki/James_Buchanan', "
                "'https://en.wikipedia.org/wiki/Harriet_Lane']"
            ),
        }
        sample = map_frames_retrieval(row)
        assert sample is not None
        assert sample.question == "Who was the 15th president?"
        assert sample.expected_uris == (
            "https://en.wikipedia.org/wiki/James_Buchanan",
            "https://en.wikipedia.org/wiki/Harriet_Lane",
        )

    def test_map_retrieval_empty_links(self) -> None:
        assert map_frames_retrieval({"Prompt": "Q", "wiki_links": "[]"}) is None

    def test_map_document_html_strips_navigation(self, tmp_path: Path) -> None:
        page = tmp_path / "article.html"
        page.write_text(
            "<html><body><p>Buchanan was a president.</p>"
            '<div role="navigation">v t e spam</div></body></html>'
        )
        row = {
            "uri": "https://en.wikipedia.org/wiki/James_Buchanan",
            "title": "James Buchanan",
            "path": str(page),
            "format": "html",
            "revid": "1364811104",
            "fetched_at": "2026-07-23",
        }
        payload = map_frames_document(row)
        assert payload.uri == "https://en.wikipedia.org/wiki/James_Buchanan"
        assert payload.title == "James Buchanan"
        assert payload.format == "html"
        assert "Buchanan was a president." in (payload.content or "")
        assert "v t e spam" not in (payload.content or "")
        assert payload.metadata == {
            "revid": "1364811104",
            "fetched_at": "2026-07-23",
        }

    def test_map_document_markdown_passthrough(self, tmp_path: Path) -> None:
        page = tmp_path / "category.md"
        page.write_text(
            "Pages in Category:Summer Olympics in London:\n- 1908 Summer Olympics\n"
        )
        row = {
            "uri": "https://en.wikipedia.org/wiki/Category:Summer_Olympics_in_London",
            "title": "Category:Summer Olympics in London",
            "path": str(page),
            "format": "md",
            "revid": None,
            "fetched_at": "2026-07-23",
        }
        payload = map_frames_document(row)
        assert payload.format == "md"
        assert "1908 Summer Olympics" in (payload.content or "")
        assert payload.metadata == {"fetched_at": "2026-07-23"}

    def test_build_case(self) -> None:
        row = {
            "Prompt": "Who was the 15th president?",
            "Answer": "James Buchanan",
            "reasoning_types": "Multiple constraints | Temporal reasoning",
        }
        case = build_frames_case(3, row)
        assert case.name == "3"
        assert case.inputs == "Who was the 15th president?"
        assert case.expected_output == "James Buchanan"
        assert case.metadata == {
            "reasoning_types": "Multiple constraints | Temporal reasoning",
            "case_index": "3",
        }

    def test_fetch_article_cache_hit_needs_no_network(self, tmp_path: Path) -> None:
        uri = "https://en.wikipedia.org/wiki/James_Buchanan"
        from urllib.parse import quote

        base = quote(uri, safe="")
        (tmp_path / f"{base}.html").write_text("<html><body>cached</body></html>")
        (tmp_path / f"{base}.json").write_text(
            '{"uri": "https://en.wikipedia.org/wiki/James_Buchanan",'
            ' "title": "James Buchanan", "format": "html",'
            ' "revid": "123", "fetched_at": "2026-07-23"}'
        )
        row = fetch_article(uri, tmp_path, client=None)
        assert row is not None
        assert row["uri"] == uri
        assert row["revid"] == "123"
        assert row["format"] == "html"
        assert Path(row["path"]).read_text().startswith("<html>")

    def test_fetch_article_category_synthesizes_members(self, tmp_path: Path) -> None:
        class StubResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return self._payload

        class StubClient:
            def get(self, url: str, params: dict | None = None) -> StubResponse:
                assert params is not None
                assert params["list"] == "categorymembers"
                return StubResponse(
                    {
                        "query": {
                            "categorymembers": [
                                {"title": "1908 Summer Olympics"},
                                {"title": "2012 Summer Olympics"},
                            ]
                        }
                    }
                )

        uri = "https://en.wikipedia.org/wiki/Category:Summer_Olympics_in_London"
        row = fetch_article(
            uri,
            tmp_path,
            client=StubClient(),  # ty: ignore[invalid-argument-type]
        )
        assert row is not None
        assert row["format"] == "md"
        content = Path(row["path"]).read_text()
        assert "1908 Summer Olympics" in content
        assert "2012 Summer Olympics" in content

    def test_fetch_article_retries_transient_failures(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(
            "evaluations.datasets.frames.time.sleep", lambda s: sleeps.append(s)
        )

        class FlakyResponse:
            text = "<html><body>ok</body></html>"
            headers = {"etag": 'W/"42/uuid"'}

            def raise_for_status(self) -> None:
                pass

        class FlakyClient:
            def __init__(self) -> None:
                self.calls = 0

            def get(self, url: str, params: dict | None = None) -> FlakyResponse:
                self.calls += 1
                if self.calls < 3:
                    raise OSError("connection reset")
                return FlakyResponse()

        client = FlakyClient()
        row = fetch_article(
            "https://en.wikipedia.org/wiki/Capybara",
            tmp_path,
            client=client,  # ty: ignore[invalid-argument-type]
        )
        assert row is not None
        assert row["revid"] == "42"
        assert client.calls == 3
        # One throttle sleep before fetching plus one backoff per failure.
        assert len(sleeps) == 3

    def test_fetch_article_honors_retry_after_on_rate_limit(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import httpx

        sleeps: list[float] = []
        monkeypatch.setattr(
            "evaluations.datasets.frames.time.sleep", lambda s: sleeps.append(s)
        )
        request = httpx.Request("GET", "https://en.wikipedia.org/x")

        class OkResponse:
            text = "<html><body>ok</body></html>"
            headers = {"etag": 'W/"42/uuid"'}

            def raise_for_status(self) -> None:
                pass

        class RateLimitedClient:
            def __init__(self) -> None:
                self.calls = 0

            def get(self, url: str, params: dict | None = None) -> OkResponse:
                self.calls += 1
                if self.calls == 1:
                    raise httpx.HTTPStatusError(
                        "429 too many requests",
                        request=request,
                        response=httpx.Response(
                            429, headers={"retry-after": "13"}, request=request
                        ),
                    )
                return OkResponse()

        row = fetch_article(
            "https://en.wikipedia.org/wiki/Capybara",
            tmp_path,
            client=RateLimitedClient(),  # ty: ignore[invalid-argument-type]
        )
        assert row is not None
        assert 13.0 in sleeps

    def test_fetch_article_gives_up_after_max_attempts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr("evaluations.datasets.frames.time.sleep", lambda s: None)

        class DeadClient:
            def __init__(self) -> None:
                self.calls = 0

            def get(self, url: str, params: dict | None = None):
                self.calls += 1
                raise OSError("connection reset")

        client = DeadClient()
        row = fetch_article(
            "https://en.wikipedia.org/wiki/Capybara",
            tmp_path,
            client=client,  # ty: ignore[invalid-argument-type]
        )
        assert row is None
        assert client.calls == FETCH_ATTEMPTS

    def test_load_corpus_raises_on_partial_fetch(self, monkeypatch) -> None:
        import evaluations.datasets.frames as frames

        monkeypatch.setattr(frames, "_cached_corpus", None)
        monkeypatch.setattr(
            frames,
            "load_frames_questions",
            lambda: [
                {
                    "wiki_links": "['https://en.wikipedia.org/wiki/A', "
                    "'https://en.wikipedia.org/wiki/B']"
                }
            ],
        )
        monkeypatch.setattr(
            frames, "fetch_article", lambda uri, cache_dir, client: None
        )
        with pytest.raises(RuntimeError, match="0/2"):
            frames.load_frames_corpus()

    def test_question_with_deleted_article_is_excluded(self) -> None:
        gone = {
            "wiki_links": "['https://en.wikipedia.org/wiki/Jack_Vance_(tennis)', "
            "'https://en.wikipedia.org/wiki/Capybara']"
        }
        kept = {"wiki_links": "['https://en.wikipedia.org/wiki/Capybara']"}
        assert question_is_answerable(gone) is False
        assert question_is_answerable(kept) is True

    def test_questions_carry_stable_ids(self, monkeypatch) -> None:
        import evaluations.datasets.frames as frames
        from datasets import Dataset

        rows = Dataset.from_list(
            [
                {
                    "Unnamed: 0": 7,
                    "wiki_links": "['https://en.wikipedia.org/wiki/Capybara']",
                },
                {
                    "Unnamed: 0": 8,
                    "wiki_links": "['https://en.wikipedia.org/wiki/Jack_Vance_(tennis)']",
                },
            ]
        )
        monkeypatch.setattr(frames, "load_frames_test", lambda: rows)
        questions = frames.load_frames_questions()
        assert [row["id"] for row in questions] == ["7"]
