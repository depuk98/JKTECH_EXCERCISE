import os
import json
import tempfile
import asyncio
import datetime
import logging
from typing import List, Dict, Optional, Tuple, BinaryIO, Any
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import func, or_, select, text
from fastapi import UploadFile, HTTPException, status
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.models.user import User
from app.schemas.document import DocumentCreate, DocumentUpdate, DocumentChunkCreate
from app.db.session import SessionLocal  # Import the session factory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model (this happens once when the service is imported)
MODEL_NAME = "all-MiniLM-L6-v2"  # Small but effective embedding model
EMBEDDING_DIM = 384  # Dimension of embeddings for this model
embedding_model = SentenceTransformer(MODEL_NAME)

class DocumentService:
    """Service for document operations."""
    
    @staticmethod
    async def upload_document(
        db: Session, 
        user: User, 
        file: UploadFile
    ) -> Document:
        """
        Upload a document file and create initial database entry.
        
        Args:
            db: Database session
            user: User object
            file: Uploaded file
            
        Returns:
            Document: The created document record
        """
        # Verify user exists
        if not user:
            raise ValueError("User not found")
        
        # Create document record
        document_in = DocumentCreate(
            user_id=user.id,
            filename=file.filename,
            content_type=file.content_type,
            file_size=file.size if hasattr(file, 'size') else None,
            status="pending"  # Status is already defined with default in the schema
        )
        
        document = Document(
            user_id=document_in.user_id,
            filename=document_in.filename,
            content_type=document_in.content_type,
            file_size=document_in.file_size,
            status=document_in.status
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Save document ID for the background task
        document_id = document.id
        
        # Create a copy of the file for processing
        # This is necessary because the file might be closed after the request is completed
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                # Reset file position
                await file.seek(0)
                
                # Write file content to temp file
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
        
            # Initiate the document processing in the background
            asyncio.create_task(
                DocumentService._process_document(document_id, temp_file_path, file.content_type)
            )
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            logger.error(f"Error preparing document for processing: {str(e)}")
            
            # Update document status to error
            document.status = "error"
            db.commit()
            
            raise e
        
        return document
    
    @staticmethod
    async def _process_document(document_id: int, temp_file_path: str, content_type: str) -> None:
        """
        Process a document asynchronously in a background task.
        
        Args:
            document_id: ID of the document to process
            temp_file_path: Path to the temporary file
            content_type: Content type of the file
        """
        # Create a new database session for the background task
        db = SessionLocal()
        
        try:
            # Get the document from the database
            document = db.query(Document).filter(Document.id == document_id).first()
            
            if not document:
                logger.error(f"Document not found: {document_id}")
                return
            
            # Update status to processing
            document.status = "processing"
            db.commit()
            
            # Extract text based on file type
            text_chunks = []
            
            try:
                if content_type == "application/pdf":
                    text_chunks = await DocumentService._load_pdf(temp_file_path)
                elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                    text_chunks = await DocumentService._load_docx(temp_file_path)
                elif content_type == "text/plain":
                    text_chunks = await DocumentService._load_text(temp_file_path)
                else:
                    raise ValueError(f"Unsupported file type: {content_type}")
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                # Process chunks
                if text_chunks:
                    # Update document page count if available
                    if hasattr(text_chunks[0], 'metadata') and 'page' in text_chunks[0].metadata:
                        document.page_count = max(chunk.metadata.get('page', 0) for chunk in text_chunks)
                    
                    # Chunk documents
                    chunks = await DocumentService._chunk_documents(text_chunks)
                    
                    # Generate embeddings and store chunks
                    for i, chunk in enumerate(chunks):
                        # Generate embedding
                        embedding = await DocumentService._generate_embeddings(chunk.page_content)
                        
                        # Create chunk in database
                        db_chunk = DocumentChunk(
                            document_id=document.id,
                            chunk_index=i,
                            text=chunk.page_content,
                            embedding=embedding,
                            chunk_metadata=chunk.metadata
                        )
                        
                        db.add(db_chunk)
                    
                    # Update document status
                    document.status = "processed"
                    db.commit()
                    logger.info(f"Document processed successfully: {document.id}")
                else:
                    # No text content found
                    document.status = "error"
                    db.commit()
                    logger.error(f"No text content found in document: {document.filename}")
            except Exception as e:
                # Update document status to error
                document.status = "error"
                db.commit()
                
                # Clean up temp file if it still exists
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
                logger.error(f"Error processing document content {document.id}: {str(e)}")
                
        except Exception as e:
            # Try to update document status to error if possible
            try:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.status = "error"
                    db.commit()
            except:
                pass
                
            logger.error(f"Error processing document {document_id}: {str(e)}")
        finally:
            # Always close the database session
            db.close()
    
    @staticmethod
    async def _load_pdf(filepath: str) -> List[Any]:
        """
        Load a PDF document.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of document pages
        """
        loader = PyPDFLoader(filepath)
        return loader.load()
    
    @staticmethod
    async def _load_docx(filepath: str) -> List[Any]:
        """
        Load a DOCX document.
        
        Args:
            filepath: Path to the DOCX file
            
        Returns:
            List of document pages
        """
        loader = Docx2txtLoader(filepath)
        return loader.load()
    
    @staticmethod
    async def _load_text(filepath: str) -> List[Any]:
        """
        Load a text document.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            List of document pages
        """
        loader = TextLoader(filepath)
        return loader.load()
    
    @staticmethod
    async def _chunk_documents(documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks using a hierarchical chunking strategy.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        # Hierarchical text splitter with overlapping windows
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        return text_splitter.split_documents(documents)
    
    @staticmethod
    async def _generate_embeddings(text: str) -> List[float]:
        """
        Generate embeddings for a text chunk.
        
        Args:
            text: Text chunk to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: embedding_model.encode(text).tolist())
    
    @staticmethod
    async def get_document_by_id(db: Session, document_id: int) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            db: Database session
            document_id: ID of the document to retrieve
            
        Returns:
            Document or None if not found
        """
        return db.query(Document).filter(Document.id == document_id).first()
    
    @staticmethod
    async def get_documents_by_user_id(
        db: Session, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 100
    ) -> Tuple[List[Document], int]:
        """
        Get all documents for a user.
        
        Args:
            db: Database session
            user_id: ID of the user
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            Tuple containing list of documents and total count
        """
        # Get total count
        total = db.query(func.count(Document.id)).filter(Document.user_id == user_id).scalar() or 0
        
        # Get documents
        documents = db.query(Document).filter(Document.user_id == user_id).order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
        
        return documents, total
    
    @staticmethod
    async def search_documents(
        db: Session,
        user_id: int,
        query: str,
        limit: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for document chunks by semantic similarity.
        
        Args:
            db: Database session
            user_id: ID of the user
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of document chunks with similarity scores
        """
        logger.info(f"Searching documents for user {user_id} with query: {query}")
        
        # Generate embedding for query
        query_embedding = await DocumentService._generate_embeddings(query)
        
        # Try vector search using pgvector
        try:
            logger.info("Attempting vector search with pgvector")
            
            # Approach 1: Using a more secure SQL approach with proper binding
            # First register a custom function that safely converts our embedding to a vector
            from sqlalchemy import func, text
            
            # Create a raw SQL with proper parameter binding
            vector_search_query = """
                WITH query_vector AS (
                    SELECT pgvector_parse_array(:embedding_json) AS vector
                )
                SELECT 
                    dc.id, 
                    dc.document_id, 
                    dc.chunk_index, 
                    dc.text, 
                    dc.embedding,
                    dc.chunk_metadata, 
                    dc.created_at,
                    1 - (dc.embedding <=> (SELECT vector FROM query_vector)) AS similarity_score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = :user_id
                ORDER BY dc.embedding <=> (SELECT vector FROM query_vector)
                LIMIT :limit
            """
            
            # Create a custom SQL function if it doesn't exist
            setup_sql = """
            CREATE OR REPLACE FUNCTION pgvector_parse_array(arr text) RETURNS vector AS $$
            BEGIN
                RETURN arr::vector;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            """
            
            # First, ensure our helper function exists
            db.execute(text(setup_sql))
            db.commit()
            
            # Now run the actual query
            logger.info("Executing vector search query")
            result = db.execute(
                text(vector_search_query), 
                {
                    "user_id": user_id,
                    "embedding_json": f"[{','.join(str(x) for x in query_embedding)}]",
                    "limit": limit
                }
            )
            
            chunks_with_scores = []
            for row in result:
                chunk = DocumentChunk(
                    id=row.id,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    chunk_metadata=row.chunk_metadata,
                    created_at=row.created_at
                )
                # Get the similarity score from the query result
                similarity_score = float(row.similarity_score)
                chunks_with_scores.append((chunk, similarity_score))
            
            if chunks_with_scores:
                logger.info(f"Vector search found {len(chunks_with_scores)} results")
                return chunks_with_scores
                
            logger.info("Vector search returned no results, falling back to keyword search")
        
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            logger.info("Falling back to keyword search")
            # Important: rollback the transaction to prevent cascading errors
            db.rollback()
        
        # Fall back to keyword search if vector search fails
        try:
            # Extract keywords from the query
            keywords = query.split()
            conditions = []
            
            # Build a query that looks for any of the keywords
            if keywords:
                for keyword in keywords:
                    if len(keyword) > 3:  # Only use keywords with more than 3 chars
                        conditions.append(f"dc.text ILIKE '%{keyword}%'")
            
            if conditions:
                condition_str = " OR ".join(conditions)
                text_search_query = f"""
                    SELECT dc.*
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = :user_id AND ({condition_str})
                    LIMIT :limit
                """
                
                logger.info(f"Executing keyword search with query: {text_search_query}")
                result = db.execute(text(text_search_query), {"user_id": user_id, "limit": limit})
                
                chunks = []
                for row in result:
                    chunk = DocumentChunk(
                        id=row.id,
                        document_id=row.document_id,
                        chunk_index=row.chunk_index,
                        text=row.text,
                        chunk_metadata=row.chunk_metadata,
                        created_at=row.created_at
                    )
                    chunks.append(chunk)
                
                # Calculate simple relevance scores based on keyword frequency
                chunks_with_scores = []
                for chunk in chunks:
                    score = 0
                    for keyword in keywords:
                        if len(keyword) > 3:
                            # Count occurrences (case insensitive)
                            score += chunk.text.lower().count(keyword.lower()) * 0.1
                    
                    # Ensure a minimum score for matches
                    score = max(score, 0.5)
                    chunks_with_scores.append((chunk, float(score)))
                
                # Sort by score
                chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                if chunks_with_scores:
                    logger.info(f"Keyword search found {len(chunks_with_scores)} results")
                    return chunks_with_scores[:limit]
        
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            # Rollback the transaction to prevent cascading errors
            db.rollback()
        
        # As a last resort, just return the most recent document chunks
        try:
            logger.info("Falling back to retrieving most recent chunks")
            
            recent_chunks_query = """
                SELECT dc.*
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = :user_id
                ORDER BY dc.created_at DESC
                LIMIT :limit
            """
            
            result = db.execute(text(recent_chunks_query), {"user_id": user_id, "limit": limit})
            
            chunks_with_scores = []
            for row in result:
                chunk = DocumentChunk(
                    id=row.id,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    chunk_metadata=row.chunk_metadata,
                    created_at=row.created_at
                )
                # Use a default score
                chunks_with_scores.append((chunk, 0.5))
            
            logger.info(f"Retrieved {len(chunks_with_scores)} most recent chunks as fallback")
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"All search fallbacks failed: {str(e)}")
            # Final rollback to ensure a clean state
            db.rollback()
            return []  # Return empty results if everything fails 

    @staticmethod
    async def delete_document(
        db: Session,
        document_id: int,
        user_id: int
    ) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            db: Database session
            document_id: ID of the document to delete
            user_id: ID of the user making the request (for verification)
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Find the document
            document = db.query(Document).filter(Document.id == document_id).first()
            
            # Check if document exists and belongs to the user
            if not document:
                logger.error(f"Document not found: {document_id}")
                return False
                
            if document.user_id != user_id:
                logger.error(f"User {user_id} attempted to delete document {document_id} belonging to user {document.user_id}")
                return False
            
            # Delete associated document chunks first (cascading will handle this automatically if set)
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            
            # Delete the document
            db.delete(document)
            db.commit()
            
            logger.info(f"Document {document_id} successfully deleted by user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False 