import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox


@pytest.fixture
async def sandbox(temp_db_path):
    """Create a Monty sandbox for testing."""
    async with HaikuRAG(temp_db_path, create=True):
        config = AppConfig()
        context = AnalysisContext()
        return Sandbox(db_path=temp_db_path, config=config, context=context)
