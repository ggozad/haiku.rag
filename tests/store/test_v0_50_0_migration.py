import json

import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades.v0_50_0 import _apply_canonical_metadata_keys
from tests.store.legacy_documents import (
    LegacyDocumentRecord,
    seed_legacy_documents,
)


@pytest.mark.asyncio
class TestV0_50_0Migration:
    """v0.50.0 normalises document.metadata to source-agnostic keys.

    Applied in isolation against the pre-0.57 documents schema (metadata still
    inline), so the assertions read documents.metadata directly. The full chain
    (where v0.57.0 later relocates metadata to document_meta) is covered by the
    v0.57.0 migration test.
    """

    async def test_renames_etag_and_content_type(self, temp_db_path):
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    LegacyDocumentRecord(
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
                    LegacyDocumentRecord(
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
                ],
            )

            await _apply_canonical_metadata_keys(store)

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
            await seed_legacy_documents(
                store,
                [
                    LegacyDocumentRecord(
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
                ],
            )

            await _apply_canonical_metadata_keys(store)
            rows = await store.documents_table.query().to_list()
            assert json.loads(rows[0]["metadata"]) == {
                "source_revision": "abc",
                "content_type": "text/plain",
                "md5": "m",
            }

    async def test_like_filter_ignores_non_key_occurrences(self, temp_db_path):
        """The WHERE short-circuit on `metadata LIKE '%"etag"%'` looks for the
        quoted-key form. Values that happen to contain the substring `etag` and
        composite key names like `my_etag_key` must not get rewritten."""
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    # `etag` only appears as a VALUE — no `etag` key.
                    LegacyDocumentRecord(
                        id="value-only",
                        content="x",
                        uri="u1",
                        metadata=json.dumps(
                            {
                                "description": "the etag of the file",
                                "source_revision": "v1",
                            }
                        ),
                    ),
                    # A composite key containing `etag` but not equal to it.
                    LegacyDocumentRecord(
                        id="composite-key",
                        content="x",
                        uri="u2",
                        metadata=json.dumps(
                            {
                                "my_etag_key": "v",
                                "source_revision": "v2",
                            }
                        ),
                    ),
                ],
            )

            await _apply_canonical_metadata_keys(store)
            rows = await store.documents_table.query().to_list()
            by_id = {r["id"]: json.loads(r["metadata"]) for r in rows}

            assert by_id["value-only"] == {
                "description": "the etag of the file",
                "source_revision": "v1",
            }
            assert by_id["composite-key"] == {
                "my_etag_key": "v",
                "source_revision": "v2",
            }

    async def test_unparseable_metadata_skipped_without_crashing(self, temp_db_path):
        """A row with malformed JSON in `metadata` must not abort the whole
        migration: it's logged and skipped, and well-formed rows alongside it
        still get rewritten. The bad row's metadata is left exactly as-is."""
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    # Contains the substring `"etag"` so the WHERE LIKE pulls it
                    # in, but it's not valid JSON — json.loads fails.
                    LegacyDocumentRecord(
                        id="bad",
                        content="x",
                        uri="u1",
                        metadata='{"etag": broken',
                    ),
                    LegacyDocumentRecord(
                        id="good",
                        content="x",
                        uri="u2",
                        metadata=json.dumps({"etag": "abc"}),
                    ),
                ],
            )

            # Must not raise.
            await _apply_canonical_metadata_keys(store)

            rows = await store.documents_table.query().to_list()
            by_id = {r["id"]: r["metadata"] for r in rows}

            assert by_id["bad"] == '{"etag": broken'
            assert json.loads(by_id["good"]) == {"source_revision": "abc"}

    async def test_preserves_existing_canonical_keys_on_collision(self, temp_db_path):
        """If both legacy and canonical keys are present, the canonical wins
        and the legacy is dropped — defends against partial-migration states."""
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    LegacyDocumentRecord(
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
                ],
            )

            await _apply_canonical_metadata_keys(store)
            rows = await store.documents_table.query().to_list()
            meta = json.loads(rows[0]["metadata"])
            assert meta == {
                "source_revision": "canonical",
                "content_type": "text/new",
            }
