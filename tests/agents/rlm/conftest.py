import pytest

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.agents.rlm.sandbox import REPLEnvironment
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import RLMConfig


@pytest.fixture
async def empty_client(temp_db_path):
    """Create an empty HaikuRAG client without documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        yield client


@pytest.fixture
async def repl_env_empty(empty_client):
    """Create a REPL environment without documents."""
    config = RLMConfig()
    context = RLMContext()
    return REPLEnvironment(client=empty_client, config=config, context=context)
