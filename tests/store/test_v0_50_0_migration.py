import json

import pytest

from haiku.rag.store.engine import DocumentRecord, Store


@pytest.mark.asyncio
class TestV0_50_0Migration:
    """v0.50.0 normalises document.metadata to source-agnostic keys."""

    async def test_renames_etag_and_content_type(self, temp_db_path):
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.48.1")
            await store.documents_table.add(
                [
                    DocumentRecord(
                        id="doc-s3",
                        content="x",
                        uri="s3://b/k",
                        metadata=json.dumps(
                            {
                                "etag": "abc123",
                                "contentType": "application/pdf",
                                "md5": "deadbeef",
                            }
                        ),
                    ),
                    DocumentRecord(
                        id="doc-fs",
                        content="y",
                        uri="file:///tmp/x.md",
                        metadata=json.dumps(
                            {
                                "contentType": "text/markdown",
                                "md5": "feedface",
                            }
                        ),
                    ),
                ]
            )

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert any("0.50.0" in d for d in applied)

            rows = await store.documents_table.query().to_list()
            by_id = {r["id"]: json.loads(r["metadata"]) for r in rows}

            assert by_id["doc-s3"] == {
                "source_revision": "abc123",
                "content_type": "application/pdf",
                "md5": "deadbeef",
            }
            assert by_id["doc-fs"] == {
                "content_type": "text/markdown",
                "md5": "feedface",
            }

    async def test_idempotent_on_already_migrated(self, temp_db_path):
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.48.1")
            await store.documents_table.add(
                [
                    DocumentRecord(
                        id="d",
                        content="x",
                        uri="s3://b/k",
                        metadata=json.dumps(
                            {
                                "source_revision": "abc",
                                "content_type": "text/plain",
                                "md5": "m",
                            }
                        ),
                    )
                ]
            )

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            rows = await store.documents_table.query().to_list()
            assert json.loads(rows[0]["metadata"]) == {
                "source_revision": "abc",
                "content_type": "text/plain",
                "md5": "m",
            }

    async def test_preserves_existing_canonical_keys_on_collision(self, temp_db_path):
        """If both legacy and canonical keys are present, the canonical wins
        and the legacy is dropped — defends against partial-migration states."""
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.48.1")
            await store.documents_table.add(
                [
                    DocumentRecord(
                        id="d",
                        content="x",
                        uri="s3://b/k",
                        metadata=json.dumps(
                            {
                                "etag": "legacy",
                                "source_revision": "canonical",
                                "contentType": "text/old",
                                "content_type": "text/new",
                            }
                        ),
                    )
                ]
            )

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            rows = await store.documents_table.query().to_list()
            meta = json.loads(rows[0]["metadata"])
            assert meta == {
                "source_revision": "canonical",
                "content_type": "text/new",
            }
