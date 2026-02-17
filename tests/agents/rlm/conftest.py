import pytest

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.agents.rlm.sandbox import Sandbox
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig


@pytest.fixture
async def empty_client(temp_db_path):
    """Create an empty HaikuRAG client without documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        yield client


@pytest.fixture
async def sandbox(empty_client):
    """Create a Monty sandbox for testing."""
    config = AppConfig()
    context = RLMContext()
    async with Sandbox(client=empty_client, config=config, context=context) as sandbox:
        yield sandbox
