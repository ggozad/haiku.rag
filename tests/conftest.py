import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, load_dataset, load_from_disk


@pytest.fixture(scope="session")
def qa_corpus() -> Dataset:
    ds_path = Path(__file__).parent / "data" / "dataset"
    ds_path.mkdir(parents=True, exist_ok=True)
    try:
        ds: Dataset = load_from_disk(ds_path)  # type: ignore
        return ds
    except FileNotFoundError:
        ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
        corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")
        corpus.save_to_disk(ds_path)
        return corpus


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test.lancedb"
