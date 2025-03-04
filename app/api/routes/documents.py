from typing import Any, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.schemas.document import Document as DocumentSchema, DocumentList
from app.services.document import DocumentService

router = APIRouter()

@router.post("/upload", response_model=DocumentSchema)
async def upload_document(
    *,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Upload a document file.
    
    The file will be processed in the background and stored in the database.
    Supported formats: PDF, DOCX, TXT
    """
    # Validate file type
    content_type = file.content_type
    valid_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain"
    ]
    
    if content_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload PDF, DOCX, or TXT files.",
        )
    
    try:
        document = await DocumentService.upload_document(
            db=db, 
            user=current_user,  # Pass the full user object, not just the ID
            file=file
        )
        return document
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@router.get("", response_model=DocumentList)
async def list_documents(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
) -> Any:
    """
    List documents for the current user.
    """
    try:
        # Call the get_documents_by_user_id method
        documents, total = await DocumentService.get_documents_by_user_id(
            db=db, 
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        
        # Log the document count for debugging
        print(f"Retrieved {len(documents)} documents for user {current_user.id} (total: {total})")
        
        # Explicitly create a DocumentList response
        response = {
            "total": total,
            "documents": documents
        }
        
        return response
    except Exception as e:
        # Log the specific error for debugging
        import logging
        logging.error(f"Error in list_documents: {str(e)}", exc_info=True)
        
        # Raise a user-friendly error
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.get("/search", response_model=List[dict])
async def search_documents(
    *,
    db: Session = Depends(get_db),
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Search for documents using semantic search.
    
    Returns a list of document chunks with similarity scores.
    """
    try:
        results = await DocumentService.search_documents(
            db=db,
            user_id=current_user.id,
            query=query,
            limit=limit
        )
        
        # Format results for response
        response = []
        for chunk, score in results:
            response.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.chunk_metadata,  # Use the chunk_metadata field
                "similarity_score": score
            })
        
        return response
    except Exception as e:
        # Log the specific error for debugging
        import logging
        logging.error(f"Error in search_documents: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to search documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(
    *,
    db: Session = Depends(get_db),
    document_id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get a document by ID.
    """
    try:
        document = await DocumentService.get_document_by_id(db=db, document_id=document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        
        if document.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )
        
        return document
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_document: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve document: {str(e)}"
        )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    *,
    db: Session = Depends(get_db),
    document_id: int,
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a document by ID.
    """
    try:
        # Attempt to delete the document
        success = await DocumentService.delete_document(
            db=db, 
            document_id=document_id, 
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or you don't have permission to delete it",
            )
        
        # For 204 No Content, don't return anything
        return None
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.error(f"Error in delete_document: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete document: {str(e)}"
        ) 