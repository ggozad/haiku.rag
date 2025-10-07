from unittest.mock import patch

import pytest

from haiku.rag.config import Config
from haiku.rag.store.engine import Store


@pytest.mark.asyncio
async def test_lancedb_cloud_skips_optimization(temp_db_path):
    """Test that vacuum is skipped when using LanceDB Cloud (db:// URI)."""
    # Create a store
    store = Store(temp_db_path)

    # Mock all cloud config to simulate LanceDB Cloud usage
    with (
        patch.object(Config, "LANCEDB_URI", "db://test-database"),
        patch.object(Config, "LANCEDB_API_KEY", "test-api-key"),
        patch.object(Config, "LANCEDB_REGION", "us-east-1"),
    ):
        # Mock the optimize method to track if it's called
        with patch.object(store.chunks_table, "optimize") as mock_optimize:
            # Call vacuum - this should skip optimization for LanceDB Cloud
            await store.vacuum()

            # The optimize method should NOT have been called for LanceDB Cloud
            mock_optimize.assert_not_called()

    store.close()


@pytest.mark.asyncio
async def test_local_storage_calls_optimization(temp_db_path):
    """Test that vacuum calls optimization for local storage."""
    # Create a store
    store = Store(temp_db_path)

    # Ensure LANCEDB_URI is empty (local storage)
    with patch.object(Config, "LANCEDB_URI", ""):
        # Mock the optimize method to track if it's called
        with patch.object(store.chunks_table, "optimize") as mock_optimize:
            # Call vacuum - this should optimize all tables for local storage
            await store.vacuum()

            # The optimize method SHOULD have been called for local storage
            mock_optimize.assert_called()

    store.close()
