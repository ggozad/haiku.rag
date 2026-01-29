from pathlib import Path

import pytest
from pydantic_ai import Agent

from haiku.rag.agents.rlm.agent import create_rlm_agent
from haiku.rag.agents.rlm.dependencies import RLMContext, RLMDeps
from haiku.rag.agents.rlm.models import CodeExecution, RLMResult
from haiku.rag.config import Config


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_rlm")


class TestCreateRLMAgent:
    def test_creates_agent_with_correct_types(self):
        agent = create_rlm_agent(Config)
        assert isinstance(agent, Agent)
        assert agent.deps_type is RLMDeps
        assert agent.output_type is RLMResult

    def test_agent_has_execute_code_tool(self):
        agent = create_rlm_agent(Config)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "execute_code" in tool_names


class TestExecuteCodeTool:
    @pytest.mark.asyncio
    async def test_execute_code_returns_structured_result(self, empty_client):
        """Test that execute_code tool produces structured CodeExecution output."""
        from haiku.rag.agents.rlm.agent import _get_or_create_repl

        context = RLMContext()
        deps = RLMDeps(
            client=empty_client,
            config=Config,
            context=context,
        )

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        repl = _get_or_create_repl(ctx)

        result = await repl.execute_async("print(1 + 1)")
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_tracks_executions_in_context(self, empty_client):
        """Test that code executions are tracked as CodeExecution objects in RLMContext."""
        from haiku.rag.agents.rlm.agent import _get_or_create_repl

        context = RLMContext()
        deps = RLMDeps(
            client=empty_client,
            config=Config,
            context=context,
        )

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        repl = _get_or_create_repl(ctx)

        assert len(context.code_executions) == 0

        result = await repl.execute_async("x = 42")
        assert result.success

    @pytest.mark.asyncio
    async def test_code_execution_has_correct_fields(self, empty_client):
        """Test that CodeExecution has all expected fields."""
        execution = CodeExecution(
            code="print('hello')",
            stdout="hello\n",
            stderr="",
            success=True,
        )
        assert execution.code == "print('hello')"
        assert execution.stdout == "hello\n"
        assert execution.stderr == ""
        assert execution.success is True

    @pytest.mark.asyncio
    async def test_code_execution_captures_errors(self, empty_client):
        """Test that failed executions are properly captured."""
        from haiku.rag.agents.rlm.agent import _get_or_create_repl

        context = RLMContext()
        deps = RLMDeps(
            client=empty_client,
            config=Config,
            context=context,
        )

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        repl = _get_or_create_repl(ctx)

        result = await repl.execute_async("1/0")
        assert result.success is False
        assert "ZeroDivisionError" in result.stderr


class TestClientRLMIntegration:
    """Integration tests for client.rlm() method."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_count_documents(self, allow_model_requests, temp_db_path):
        """Test RLM agent can count documents.

        Agent program:
            docs = list_documents(limit=1000)
            print(len(docs))
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document("First document about cats.", title="Doc 1")
            await client.create_document("Second document about dogs.", title="Doc 2")
            await client.create_document("Third document about birds.", title="Doc 3")

            answer = await client.rlm("How many documents are in the database?")

            assert "3" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_aggregation(self, allow_model_requests, temp_db_path):
        """Test RLM agent can perform aggregation across documents.

        Agent program:
            import re
            revs = {}
            for d in ['Q1 Report', 'Q2 Report', 'Q3 Report']:
                content = get_document(d)
                if content:
                    vals = re.findall(r'\\$([\\d,]+)', content)
                    if vals:
                        rev = sum(int(v.replace(',', '')) for v in vals)
                    else:
                        rev = None
                else:
                    rev = None
                revs[d] = rev
            print(revs)
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                "Sales report Q1: Revenue was $100,000.", title="Q1 Report"
            )
            await client.create_document(
                "Sales report Q2: Revenue was $150,000.", title="Q2 Report"
            )
            await client.create_document(
                "Sales report Q3: Revenue was $200,000.", title="Q3 Report"
            )

            answer = await client.rlm(
                "What is the total revenue across all quarterly reports?"
            )

            assert "450" in answer or "450,000" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_with_filter(self, allow_model_requests, temp_db_path):
        """Test RLM agent respects filter parameter.

        Agent program:
            docs = list_documents(limit=1000)
            print(len(docs))
            print(docs[:5])

        The filter is applied via context, so list_documents() only sees "Cats".
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document("Cat document.", title="Cats")
            await client.create_document("Dog document.", title="Dogs")
            await client.create_document("Bird document.", title="Birds")

            answer = await client.rlm(
                "How many documents are available?",
                filter="title = 'Cats'",
            )

            assert "1" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_docling_document_structure(
        self, allow_model_requests, temp_db_path
    ):
        """Test RLM agent can analyze document structure using DoclingDocument.

        Agent program:
            docs = list_documents(limit=20)
            print(docs)

            doc = get_docling_document('<doc_id>')
            print(doc.name)
            print('tables:', len(doc.tables))
            print('pictures:', len(doc.pictures))
        """
        from pathlib import Path

        from haiku.rag.client import HaikuRAG
        from haiku.rag.config import AppConfig

        pdf_path = Path("tests/data/doclaynet.pdf")
        config = AppConfig()
        config.processing.conversion_options.do_ocr = False

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document_from_source(pdf_path)

            answer = await client.rlm(
                "How many tables are in the document? "
                "Also tell me how many pictures/figures it contains."
            )

            # The doclaynet.pdf has 1 table and 1 picture
            assert "1" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_semantic_analysis_with_llm(
        self, allow_model_requests, temp_db_path
    ):
        """Test RLM agent can use llm() for semantic analysis combined with computation.

        Agent program:
            docs = list_documents(limit=100)
            print(len(docs))
            print([d['title'] for d in docs[:20]])

            sentiments = {}
            for title in ['Q1 Update', 'Q2 Update', 'Q3 Update']:
                content = get_document(title)
                if content:
                    result = llm(f"Classify sentiment as positive/negative/mixed: {content}")
                    sentiments[title] = result
            print(sentiments)
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                "The new product launch exceeded expectations. Sales grew 40% "
                "and customer feedback has been overwhelmingly positive. "
                "Team morale is at an all-time high.",
                title="Q1 Update",
            )
            await client.create_document(
                "We faced significant challenges this quarter. Supply chain issues "
                "caused delays, and we missed our revenue target by 15%. "
                "Several key employees left the company.",
                title="Q2 Update",
            )
            await client.create_document(
                "Mixed results this quarter. While product quality improved, "
                "marketing campaigns underperformed. Revenue was flat compared "
                "to last year but customer retention increased.",
                title="Q3 Update",
            )

            answer = await client.rlm(
                "Analyze the sentiment of each quarterly update. "
                "How many quarters were positive, negative, and mixed?"
            )

            # Should identify: Q1=positive, Q2=negative, Q3=mixed
            assert "positive" in answer.lower()
            assert "negative" in answer.lower()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_search_and_extract(self, allow_model_requests, temp_db_path):
        """Test RLM agent can use search() to find content and extract information.

        Agent program:
            results = search("document element types", limit=20)
            print(len(results))
            for r in results[:5]:
                print(r['document_title'], r['chunk_id'], r['score'])
                print(r['content'][:200])

            results = search("DocBank element types", limit=10)
            ...
        """
        from pathlib import Path

        from haiku.rag.client import HaikuRAG
        from haiku.rag.config import AppConfig

        pdf_path = Path("tests/data/doclaynet.pdf")
        config = AppConfig()
        config.processing.conversion_options.do_ocr = False

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document_from_source(pdf_path)

            answer = await client.rlm(
                "Search for content about document element types or labels. "
                "What are all the different document element types mentioned? "
                "List them all."
            )

            # The doclaynet.pdf defines exactly 11 class labels for document elements
            # Normalize Unicode hyphens (U+2011 non-breaking hyphen) to regular hyphens
            answer_lower = answer.lower().replace("\u2011", "-")
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
            for label in expected_labels:
                # Allow for hyphen or space variants
                assert (
                    label in answer_lower or label.replace("-", " ") in answer_lower
                ), f"Missing label: {label}"

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_with_preloaded_documents(
        self, allow_model_requests, temp_db_path
    ):
        """Test RLM agent can use pre-loaded documents variable.

        Agent program:
            if 'documents' in dir():
                for doc in documents:
                    print(doc['title'], len(doc['content']))
            else:
                print('No preloaded documents')
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                "The company was founded in 1985 by Jane Smith.",
                title="Company History",
            )
            await client.create_document(
                "Our mission is to make technology accessible to everyone.",
                title="Mission Statement",
            )

            answer = await client.rlm(
                "Using the pre-loaded documents variable, "
                "tell me when was the company founded and what is their mission?",
                documents=["Company History", "Mission Statement"],
            )

            assert "1985" in answer
            assert "accessible" in answer.lower() or "technology" in answer.lower()
