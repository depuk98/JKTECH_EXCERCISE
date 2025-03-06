from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.db.session import get_async_db
from app.models.user import User
from app.api.deps import get_current_active_user_async
from app.services.rag import rag_service
from app.services.document import DocumentService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ask", response_model=dict)
async def ask_question(
    *,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user_async),
    query: str = Body(None, embed=True),
    question: str = Body(None, embed=True),
    document_ids: Optional[List[int]] = Body(None, embed=True),
    use_cache: bool = Body(True, embed=True)
) -> Any:
    """
    Ask a question about documents using RAG-based Q&A.
    
    This endpoint takes a natural language query and generates an answer based on 
    information from the user's documents. If document_ids are provided, only those
    specific documents will be used to answer the question.
    
    Args:
        query: The user's question
        question: Alternative parameter name for the user's question (for backwards compatibility)
        document_ids: Optional list of document IDs to search within
        use_cache: Whether to use cached responses
        
    Returns:
        Dictionary with answer, citations, and metadata
    """
    try:
        # Use query parameter or fall back to question parameter
        actual_query = query or question
        
        if not actual_query or len(actual_query.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Validate document_ids if provided
        if document_ids:
            for doc_id in document_ids:
                document = await DocumentService.get_document_by_id(db, doc_id)
                
                # Check if document exists and belongs to the user
                if not document:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document with ID {doc_id} not found"
                    )
                
                if document.user_id != current_user.id:
                    raise HTTPException(
                        status_code=403,
                        detail=f"You don't have access to document with ID {doc_id}"
                    )
                
                # Check if document is processed
                if document.status != "processed":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Document with ID {doc_id} is not fully processed yet"
                    )
        
        # Get the answer from the RAG service
        result = await rag_service.answer_question(
            db=db,
            user_id=current_user.id,
            query=actual_query,
            document_ids=document_ids,
            use_cache=use_cache
        )
        
        # Check if there was an error with Ollama
        if "metadata" in result and "error" in result["metadata"] and "Ollama" in result["metadata"]["error"]:
            logger.warning(f"Ollama service error: {result['metadata']['error']}")
            # Return a 503 Service Unavailable with the error message
            raise HTTPException(
                status_code=503,
                detail=result["answer"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )

