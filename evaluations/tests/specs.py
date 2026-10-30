from evaluations.config import DatasetSpec


def stub_spec(**overrides) -> DatasetSpec:
    """A DatasetSpec with inert loaders and mappers, any field overridden."""
    fields: dict = {
        "key": "test",
        "db_filename": "test.lancedb",
        "document_loader": lambda: None,
        "document_mapper": lambda doc: None,
        "qa_loader": lambda: [],
        "qa_case_builder": lambda idx, doc: None,
    }
    return DatasetSpec(**{**fields, **overrides})
