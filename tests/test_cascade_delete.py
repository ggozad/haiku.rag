from haiku.rag.app import HaikuRAGApp
from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import parent_uri_filter
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.document import Document


def test_parent_uri_filter_simple():
    f = parent_uri_filter("file:///path/to/parent.pdf")
    assert f == 'metadata LIKE \'%"parent_uri": "file:///path/to/parent.pdf"%\''


def test_parent_uri_filter_escapes_single_quote():
    f = parent_uri_filter("file:///x's.pdf")
    assert "''" in f


def test_parent_uri_filter_escapes_backslash():
    f = parent_uri_filter("file:///x\\y.pdf")
    assert "\\\\" in f


async def test_delete_cascades_to_children(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///path/to/parent.pdf"
        parent = await client.document_repository.create(
            Document(content="parent body", uri=parent_uri, metadata={})
        )
        child_a = await client.document_repository.create(
            Document(
                content="child A body",
                uri=f"{parent_uri}#attachment=a.pdf",
                metadata={"parent_uri": parent_uri},
            )
        )
        child_b = await client.document_repository.create(
            Document(
                content="child B body",
                uri=f"{parent_uri}#attachment=b.pdf",
                metadata={"parent_uri": parent_uri},
            )
        )

        deleted = await client.delete_document(parent.id)
        assert deleted is True

        assert await client.get_document_by_id(parent.id) is None
        assert await client.get_document_by_id(child_a.id) is None
        assert await client.get_document_by_id(child_b.id) is None


async def test_delete_leaves_unrelated_documents(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///path/to/parent.pdf"
        parent = await client.document_repository.create(
            Document(content="parent", uri=parent_uri, metadata={})
        )
        child = await client.document_repository.create(
            Document(
                content="child",
                uri=f"{parent_uri}#attachment=a.pdf",
                metadata={"parent_uri": parent_uri},
            )
        )
        unrelated = await client.document_repository.create(
            Document(
                content="unrelated",
                uri="file:///path/to/other.pdf",
                metadata={},
            )
        )

        await client.delete_document(parent.id)

        assert await client.get_document_by_id(child.id) is None
        survivor = await client.get_document_by_id(unrelated.id)
        assert survivor is not None
        assert survivor.id == unrelated.id


async def test_delete_cascades_recursively(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        gp_uri = "file:///path/to/grandparent.pdf"
        parent_uri = f"{gp_uri}#attachment=parent.pdf"

        grandparent = await client.document_repository.create(
            Document(content="gp", uri=gp_uri, metadata={})
        )
        parent = await client.document_repository.create(
            Document(
                content="p",
                uri=parent_uri,
                metadata={"parent_uri": gp_uri},
            )
        )
        child = await client.document_repository.create(
            Document(
                content="c",
                uri=f"{parent_uri}#attachment=leaf.pdf",
                metadata={"parent_uri": parent_uri},
            )
        )

        await client.delete_document(grandparent.id)

        assert await client.get_document_by_id(grandparent.id) is None
        assert await client.get_document_by_id(parent.id) is None
        assert await client.get_document_by_id(child.id) is None


async def test_delete_nonexistent_returns_false(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        result = await client.delete_document("does-not-exist")
        assert result is False


async def test_delete_handles_self_referential_parent(temp_db_path):
    """A document whose metadata.parent_uri points at its own uri must not
    cascade into infinite recursion."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        uri = "file:///path/to/self.pdf"
        doc = await client.document_repository.create(
            Document(content="self-loop", uri=uri, metadata={"parent_uri": uri})
        )

        deleted = await client.delete_document(doc.id)
        assert deleted is True
        assert await client.get_document_by_id(doc.id) is None


async def test_delete_succeeds_with_embedding_dim_mismatch(temp_db_path):
    """Deletion touches no embeddings, so a vector_dim mismatch must not block it."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.document_repository.create(
            Document(content="body", uri="file:///doc.pdf", metadata={})
        )

    mismatched = AppConfig()
    mismatched.embeddings.model.vector_dim = 9999

    app = HaikuRAGApp(db_path=temp_db_path, config=mismatched)
    await app.delete_document(doc.id)

    async with HaikuRAG(
        temp_db_path, config=mismatched, skip_validation=True
    ) as client:
        assert await client.get_document_by_id(doc.id) is None


def test_processing_config_extract_pdf_attachments_default_true():
    from haiku.rag.config.models import ProcessingConfig

    assert ProcessingConfig().extract_pdf_attachments is True


def test_processing_config_extract_pdf_attachments_overridable():
    from haiku.rag.config.models import ProcessingConfig

    cfg = ProcessingConfig(extract_pdf_attachments=False)
    assert cfg.extract_pdf_attachments is False
