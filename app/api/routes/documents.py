from typing import Any, List
import logging
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user_async
from app.db.session import get_async_db
from app.models.user import User
from app.schemas.document import Document as DocumentSchema, DocumentList
from app.services.document import DocumentService
from app.utils.model_conversion import convert_document_to_dict, convert_documents_to_list_response

router = APIRouter()

@router.post("/upload", response_model=DocumentSchema)
async def upload_document(
    *,
    db: AsyncSession = Depends(get_async_db),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user_async),
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

@router.get("/", response_model=DocumentList)
async def list_documents(
    db: AsyncSession = Depends(get_async_db),
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of documents to return"),
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Retrieve documents belonging to current user.
    """
    try:
        documents, total = await DocumentService.get_documents_by_user_id(
            db=db,
            user_id=current_user.id,
            skip=skip,
            limit=limit,
        )
        
        # Convert each document to a dictionary that matches the schema
        doc_dicts = [convert_document_to_dict(doc) for doc in documents]
        
        # Use the utility function to create the response
        return {
            "total": total,
            "documents": doc_dicts
        }
    except Exception as e:
        logging.error(f"Error in list_documents: {e}")
        logging.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.get("", response_model=DocumentList)
async def list_documents_no_slash(
    db: AsyncSession = Depends(get_async_db),
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of documents to return"),
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Retrieve documents belonging to current user (endpoint without trailing slash).
    """
    try:
        documents, total = await DocumentService.get_documents_by_user_id(
            db=db,
            user_id=current_user.id,
            skip=skip,
            limit=limit,
        )
        
        # Convert each document to a dictionary that matches the schema
        doc_dicts = [convert_document_to_dict(doc) for doc in documents]
        
        # Use the utility function to create the response
        return {
            "total": total,
            "documents": doc_dicts
        }
    except Exception as e:
        logging.error(f"Error in list_documents: {e}")
        logging.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.get("/search", response_model=List[dict])
async def search_documents(
    *,
    db: AsyncSession = Depends(get_async_db),
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Search documents for the current user.
    """
    try:
        # Call search method from DocumentService
        results = await DocumentService.search_documents(
            db=db,
            user_id=current_user.id,
            query=query,
            limit=limit
        )
        
        # Create a set of unique document ids from results
        document_ids = set()
        for chunk, _ in results:
            if hasattr(chunk, 'document_id'):
                document_ids.add(chunk.document_id)
        
        # Get documents by their ids to retrieve filenames
        documents_map = {}
        for doc_id in document_ids:
            doc = await DocumentService.get_document_by_id(db, document_id=doc_id)
            if doc:
                documents_map[doc_id] = doc
        
        # Format the results
        formatted_results = []
        for chunk, score in results:
            # Extract attributes from chunk object
            chunk_dict = {}
            for attr in ["id", "document_id", "text", "chunk_metadata"]:
                if hasattr(chunk, attr):
                    chunk_dict[attr] = getattr(chunk, attr)
            
            # If we have a MagicMock without document_id, use default
            if not chunk_dict.get("document_id") and hasattr(chunk, "document_id"):
                chunk_dict["document_id"] = getattr(chunk, "document_id")
                
            document_id = chunk_dict.get("document_id", 1)
            document_filename = "Unknown"
            
            # Get the document filename if available
            if document_id in documents_map:
                document_filename = getattr(documents_map[document_id], "filename", "Unknown")
                
            formatted_result = {
                "id": chunk_dict.get("id", 0),
                "document_id": document_id,
                "document_filename": document_filename,
                "text": chunk_dict.get("text", ""),
                "metadata": chunk_dict.get("chunk_metadata", {}),
                "score": float(score)
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
    except Exception as e:
        logging.error(f"Error in search_documents: {e}")
        logging.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(
    *,
    db: AsyncSession = Depends(get_async_db),
    document_id: int,
    current_user: User = Depends(get_current_active_user_async),
) -> Any:
    """
    Get a document by ID.
    """
    try:
        # Use as_dict=True to get a dictionary directly
        document = await DocumentService.get_document_by_id(
            db=db, 
            document_id=document_id,
            as_dict=True
        )
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        
        # Since document is now a dictionary, we access the user_id directly
        if document["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )
        
        # No need to convert, document is already a dictionary
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
    db: AsyncSession = Depends(get_async_db),
    document_id: int,
    current_user: User = Depends(get_current_active_user_async),
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
        logging.error(f"Error in delete_document: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete document: {str(e)}"
        ) 