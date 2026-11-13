from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status

from haiku.rag.client.exceptions import AttachmentDeletionError, SourceOwnedError
from haiku.rag.ingester.api.schemas import DeleteResponse, DocumentEdit, EditedDocument
from haiku.rag.ingester.api.server import APIState, get_state

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

router = APIRouter(prefix="/databases/{name:path}/documents", tags=["documents"])


def _editing(state: APIState, name: str) -> AbstractAsyncContextManager["HaikuRAG"]:
    if state.editing is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no database is running",
        )
    try:
        return state.editing(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="database not ingested"
        ) from exc


@router.patch("/{document_id}", response_model=EditedDocument)
async def edit_document(
    name: str,
    document_id: str,
    edit: DocumentEdit,
    state: APIState = Depends(get_state),
) -> EditedDocument:
    """Set a document's title and its editor metadata (`metadata.user`)."""
    async with _editing(state, name) as client:
        try:
            document = await client.edit_document(
                document_id, title=edit.title, user_metadata=edit.user
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
    assert document.id is not None
    return EditedDocument(
        id=document.id,
        uri=document.uri,
        title=document.title,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    name: str, document_id: str, state: APIState = Depends(get_state)
) -> DeleteResponse:
    """Delete a document without a source, with its attachments."""
    async with _editing(state, name) as client:
        try:
            deleted = await client.delete_document(document_id, unowned_only=True)
        except (SourceOwnedError, AttachmentDeletionError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )
    return DeleteResponse(document_id=document_id, deleted=True)
