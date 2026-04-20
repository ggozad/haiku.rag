import pytest

from haiku.rag.agents.analysis.dependencies import AnalysisContext
from haiku.rag.agents.analysis.sandbox import Sandbox
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig


@pytest.fixture
async def empty_client(temp_db_path):
    """Create an empty HaikuRAG client without documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        yield client


@pytest.fixture
async def sandbox(temp_db_path):
    """Create a Monty sandbox for testing."""
    async with HaikuRAG(temp_db_path, create=True):
        config = AppConfig()
        context = AnalysisContext()
        return Sandbox(db_path=temp_db_path, config=config, context=context)
