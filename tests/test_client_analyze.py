from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_client_analyze")


class TestClientAnalysisIntegration:
    """Integration tests for client.analyze() through the rag-analysis skill."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_analyze_count_documents(self, allow_model_requests, temp_db_path):
        config = AppConfig()

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document("First document about cats.", title="Doc 1")
            await client.create_document("Second document about dogs.", title="Doc 2")
            await client.create_document("Third document about birds.", title="Doc 3")

            result = await client.analyze("How many documents are in the database?")

            assert "3" in result.answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_analyze_aggregation(self, allow_model_requests, temp_db_path):
        config = AppConfig()

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document(
                "Sales report Q1: Revenue was $100,000.", title="Q1 Report"
            )
            await client.create_document(
                "Sales report Q2: Revenue was $150,000.", title="Q2 Report"
            )
            await client.create_document(
                "Sales report Q3: Revenue was $200,000.", title="Q3 Report"
            )

            result = await client.analyze(
                "What is the total revenue across all quarterly reports?"
            )

            assert "450" in result.answer or "450,000" in result.answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_analyze_with_filter(self, allow_model_requests, temp_db_path):
        config = AppConfig()

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document("Cat document.", title="Cats")
            await client.create_document("Dog document.", title="Dogs")
            await client.create_document("Bird document.", title="Birds")

            result = await client.analyze(
                "How many documents are available?",
                filter="title = 'Cats'",
            )

            assert "1" in result.answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_analyze_search_and_identify_source(
        self, allow_model_requests, temp_db_path
    ):
        config = AppConfig()

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document(
                "The quick brown fox jumps over the lazy dog.",
                title="Animal Facts",
            )

            result = await client.analyze(
                "Search for content about animals and tell me "
                "which document it came from."
            )

            assert "Animal Facts" in result.answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_analyze_search_and_extract(self, allow_model_requests, temp_db_path):
        pdf_path = Path("tests/data/doclaynet.pdf")
        config = AppConfig()
        config.processing.conversion_options.do_ocr = False

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document_from_source(pdf_path)

            result = await client.analyze(
                "Search for content about document element types or labels. "
                "What are all the different document element types mentioned? "
                "List them all."
            )

            answer_lower = result.answer.lower().replace("‑", "-")
            expected_labels = [
                "caption",
                "footnote",
                "formula",
                "list-item",
                "page-footer",
                "page-header",
                "picture",
                "section-header",
                "table",
                "text",
                "title",
            ]
            found_labels = [
                label
                for label in expected_labels
                if label in answer_lower or label.replace("-", " ") in answer_lower
            ]
            assert len(found_labels) >= 6, (
                f"Expected at least 6 labels, found {len(found_labels)}: {found_labels}"
            )
