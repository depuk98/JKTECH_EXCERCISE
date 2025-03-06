import os
import json
import tempfile
import asyncio
import datetime
import logging
import re
from typing import List, Dict, Optional, Tuple, BinaryIO, Any, Union
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
from app.schemas.document import DocumentCreate, DocumentUpdate, DocumentChunkCreate, Document as DocumentSchema
from app.utils.model_conversion import convert_document_to_dict, sqlalchemy_to_pydantic
from app.db.session import AsyncSessionLocal  # Import the async session factory

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
        db: AsyncSession, 
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
        document_in = {
            "user_id": user.id,
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file.size if hasattr(file, 'size') else None,
            "status": "pending"  # Initial status
        }
        
        # Safely handle the document creation
        temp_file_path = None
        try:
            # Use direct SQL to insert the document
            stmt = text("""
                INSERT INTO documents (user_id, filename, content_type, file_size, status, created_at)
                VALUES (:user_id, :filename, :content_type, :file_size, :status, NOW())
                RETURNING id, user_id, filename, content_type, file_size, file_path, status, page_count, error_message, created_at, updated_at
            """)
            
            result = await db.execute(
                stmt, 
                {
                    "user_id": document_in["user_id"],
                    "filename": document_in["filename"],
                    "content_type": document_in["content_type"],
                    "file_size": document_in["file_size"],
                    "status": document_in["status"]
                }
            )
            
            # Get the inserted document data
            doc_row = result.first()
            
            # Create a Document instance with the fetched values to return
            document = Document(
                id=doc_row.id,
                user_id=doc_row.user_id,
                filename=doc_row.filename,
                content_type=doc_row.content_type,
                file_size=doc_row.file_size,
                file_path=doc_row.file_path,
                status=doc_row.status,
                page_count=doc_row.page_count,
                error_message=doc_row.error_message,
                created_at=doc_row.created_at,
                updated_at=doc_row.updated_at
            )
            
            await db.commit()
            
            # Save document ID for the background task
            document_id = document.id
            
            # Create a copy of the file for processing
            # This is necessary because the file might be closed after the request is completed
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
            
            return document
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            # Roll back the transaction in case of error
            await db.rollback()
            logger.error(f"Error uploading document: {str(e)}")
            raise
    
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
        async with AsyncSessionLocal() as db:
            try:
                # Get the document from the database using a direct SQL query to avoid lazy loading
                doc_result = await db.execute(
                    text("""
                        SELECT id, user_id, filename, content_type, file_size, file_path, 
                               status, page_count, error_message
                        FROM documents 
                        WHERE id = :document_id
                    """),
                    {"document_id": document_id}
                )
                
                doc_row = doc_result.first()
                
                if not doc_row:
                    logger.error(f"Document not found: {document_id}")
                    return
                
                # Create a Document instance with the fetched values
                document = Document(
                    id=doc_row.id,
                    user_id=doc_row.user_id,
                    filename=doc_row.filename,
                    content_type=doc_row.content_type,
                    file_size=doc_row.file_size,
                    file_path=doc_row.file_path,
                    status=doc_row.status,
                    page_count=doc_row.page_count,
                    error_message=doc_row.error_message
                )
                
                # Update status to processing
                stmt = text("""
                    UPDATE documents 
                    SET status = 'processing' 
                    WHERE id = :document_id
                """)
                await db.execute(stmt, {"document_id": document_id})
                await db.commit()
                
                # Now we can safely use the document's attributes
                filename = document.filename  # Access filename from our populated object
                logger.info(f"Starting processing for document {document_id} (filename: {filename}, type: {content_type})")
                
                # Check if file exists
                if not os.path.exists(temp_file_path):
                    logger.error(f"Temporary file not found: {temp_file_path}")
                    
                    # Update error status with direct SQL
                    stmt = text("""
                        UPDATE documents 
                        SET status = 'error', error_message = 'Temporary file not found'
                        WHERE id = :document_id
                    """)
                    await db.execute(stmt, {"document_id": document_id})
                    await db.commit()
                    return
                    
                # Log file size
                file_size = os.path.getsize(temp_file_path)
                logger.info(f"File size: {file_size} bytes")
                
                # Extract text based on file type
                text_chunks = []
                
                try:
                    if content_type == "application/pdf":
                        logger.info(f"Processing PDF document: {document.filename}")
                        text_chunks = await DocumentService._load_pdf(temp_file_path)
                        
                        # Check if this is a scanned PDF with no text
                        if text_chunks and len(text_chunks) > 0:
                            text_content = DocumentService._check_pdf_text_content(text_chunks)
                            if text_content == "empty":
                                logger.warning(f"PDF appears to be a scanned document with no extractable text: {filename}")
                                
                                # Update warning status with direct SQL
                                stmt = text("""
                                    UPDATE documents 
                                    SET status = 'warning', 
                                        error_message = 'This appears to be a scanned PDF with no extractable text. Please upload a PDF with searchable text or use OCR software first.'
                                    WHERE id = :document_id
                                """)
                                await db.execute(stmt, {"document_id": document_id})
                                await db.commit()
                                
                                # Clean up the temp file
                                if os.path.exists(temp_file_path):
                                    os.unlink(temp_file_path)
                                
                                return
                    elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                        logger.info(f"Processing DOCX document: {document.filename}")
                        text_chunks = await DocumentService._load_docx(temp_file_path)
                    elif content_type == "text/plain":
                        logger.info(f"Processing text document: {document.filename}")
                        text_chunks = await DocumentService._load_text(temp_file_path)
                    else:
                        # Default to treating as text
                        logger.info(f"Processing unknown document type as text: {document.filename} (content-type: {content_type})")
                        text_chunks = await DocumentService._load_text(temp_file_path)
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    logger.info(f"Temporary file deleted: {temp_file_path}")
                    
                    # Process chunks
                    if text_chunks:
                        logger.info(f"Extracted {len(text_chunks)} text chunks from document {document_id}")
                        
                        # Update document page count if available
                        if hasattr(text_chunks[0], 'metadata') and 'page' in text_chunks[0].metadata:
                            page_count = max(chunk.metadata.get('page', 0) for chunk in text_chunks)
                            # Update page count with direct SQL
                            stmt = text("""
                                UPDATE documents 
                                SET page_count = :page_count
                                WHERE id = :document_id
                            """)
                            await db.execute(stmt, {"document_id": document_id, "page_count": page_count})
                            await db.commit()
                            logger.info(f"Document {document_id} page count: {page_count}")
                        
                        # Chunk documents with validation
                        logger.info(f"Starting chunking for document {document_id}")
                        chunks = await DocumentService._chunk_document(text_chunks)
                        
                        if not chunks or len(chunks) == 0:
                            logger.error(f"No chunks created for document {document_id}")
                            # Update error status with direct SQL
                            stmt = text("""
                                UPDATE documents 
                                SET status = 'error', 
                                    error_message = 'Document could not be chunked properly. This may be due to an empty or encrypted PDF, or a PDF containing only images without text.'
                                WHERE id = :document_id
                            """)
                            await db.execute(stmt, {"document_id": document_id})
                            await db.commit()
                            return
                        
                        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
                        
                        # Generate embeddings and store chunks in batches for better efficiency
                        chunks_added = 0
                        
                        # Prepare batch data
                        valid_chunks = []
                        clean_contents = []
                        
                        # First pass: validate and clean chunks
                        for i, chunk in enumerate(chunks):
                            if not hasattr(chunk, 'page_content') or not chunk.page_content or not isinstance(chunk.page_content, str):
                                logger.warning(f"Invalid chunk content for document {document_id}, chunk {i}: {type(getattr(chunk, 'page_content', None))}")
                                continue
                            
                            # Clean text to remove problematic Unicode characters
                            clean_content = DocumentService._clean_text(chunk.page_content)
                            if not clean_content:
                                logger.warning(f"Empty content after cleaning for document {document_id}, chunk {i}")
                                continue
                                
                            valid_chunks.append((i, chunk, clean_content))
                            clean_contents.append(clean_content)
                        
                        if not valid_chunks:
                            logger.error(f"No valid chunks for document {document_id}")
                            # Update error status with direct SQL
                            stmt = text("""
                                UPDATE documents 
                                SET status = 'error', 
                                    error_message = 'Document could not be processed properly. No valid text content found.'
                                WHERE id = :document_id
                            """)
                            await db.execute(stmt, {"document_id": document_id})
                            await db.commit()
                            return
                        
                        # Generate embeddings in a single batch for all valid chunks
                        try:
                            logger.info(f"Generating embeddings for {len(valid_chunks)} chunks in batch")
                            embeddings = await DocumentService._generate_embeddings_batch(clean_contents)
                            
                            # Create chunks in database
                            for idx, (chunk_idx, chunk, clean_content) in enumerate(valid_chunks):
                                embedding = embeddings[idx]
                                
                                db_chunk = DocumentChunk(
                                    document_id=document.id,
                                    chunk_index=chunk_idx,
                                    text=clean_content,
                                    embedding=embedding,
                                    chunk_metadata=getattr(chunk, 'metadata', None)
                                )
                                
                                db.add(db_chunk)
                                chunks_added += 1
                                
                            await db.commit()
                            logger.info(f"Successfully added {chunks_added} chunks to database for document {document_id}")
                            
                            # Update document status to processed with direct SQL
                            stmt = text("""
                                UPDATE documents 
                                SET status = 'processed'
                                WHERE id = :document_id
                            """)
                            await db.execute(stmt, {"document_id": document_id})
                            await db.commit()
                            logger.info(f"Document processed successfully: {document_id}")
                        except Exception as e:
                            logger.error(f"Error processing chunks for document {document_id}: {e}")
                            # Update error status with direct SQL
                            stmt = text("""
                                UPDATE documents 
                                SET status = 'error', 
                                    error_message = :error_message
                                WHERE id = :document_id
                            """)
                            await db.execute(stmt, {
                                "document_id": document_id, 
                                "error_message": f"Error during embedding generation: {str(e)}"
                            })
                            await db.commit()
                    else:
                        # No text content found
                        # Update error status with direct SQL
                        stmt = text("""
                            UPDATE documents 
                            SET status = 'error', 
                                error_message = 'No valid content extracted from document. This may be due to an encrypted PDF, a PDF containing only images, or an empty document.'
                            WHERE id = :document_id
                        """)
                        await db.execute(stmt, {"document_id": document_id})
                        await db.commit()
                        logger.error(f"No text content found in document: {filename}")
                except Exception as e:
                    # Handle transaction state before updating
                    await db.rollback()
                    
                    # Update document status to error with direct SQL
                    stmt = text("""
                        UPDATE documents 
                        SET status = 'error', 
                            error_message = :error_message
                        WHERE id = :document_id
                    """)
                    await db.execute(stmt, {
                        "document_id": document_id, 
                        "error_message": str(e)
                    })
                    await db.commit()
                    
                    # Clean up temp file if it still exists
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
                    logger.error(f"Error processing document content {document_id}: {str(e)}", exc_info=True)
                    
            except Exception as e:
                # Try to update document status to error with direct SQL
                try:
                    stmt = text("""
                        UPDATE documents 
                        SET status = 'error', 
                            error_message = :error_message
                        WHERE id = :document_id
                    """)
                    await db.execute(stmt, {
                        "document_id": document_id, 
                        "error_message": str(e)
                    })
                    await db.commit()
                except Exception as inner_e:
                    logger.error(f"Error updating document status: {inner_e}")
                    
                logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
    
    @staticmethod
    async def _load_pdf(filepath: str) -> List[Any]:
        """
        Load a PDF document.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of document pages
        """
        try:
            logger.info(f"Loading PDF from {filepath}")
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            logger.info(f"PDF loaded successfully. Found {len(documents)} page(s).")
            
            # Log the first bit of content to verify text extraction
            if documents and len(documents) > 0:
                sample_content = documents[0].page_content[:100] if hasattr(documents[0], 'page_content') else "No page_content attribute"
                logger.info(f"Sample content from first page: {sample_content}")
                
                # Check if the PDF might be a scanned document with no text
                text_content = DocumentService._check_pdf_text_content(documents)
                if text_content == "empty":
                    logger.warning(f"PDF appears to be empty or contains no extractable text")
                elif text_content == "minimal":
                    logger.warning(f"PDF appears to contain very little text, might be a scanned document")
            else:
                logger.warning(f"PDF loaded, but no document pages were returned")
            
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _check_pdf_text_content(documents: List[Any]) -> str:
        """
        Check if a PDF document contains actual text or might be a scanned document.
        
        Args:
            documents: List of document pages
            
        Returns:
            String indicating content status: "normal", "minimal", or "empty"
        """
        if not documents:
            return "empty"
            
        # Calculate total text content length
        total_text = 0
        total_pages = len(documents)
        
        for doc in documents:
            if hasattr(doc, 'page_content'):
                page_text = doc.page_content.strip()
                total_text += len(page_text)
                
        logger.info(f"PDF contains {total_text} characters across {total_pages} pages")
        
        # Determine if the PDF has enough text content
        if total_text == 0:
            return "empty"
        elif total_text < 50 * total_pages:  # Heuristic: Less than 50 chars per page on average
            return "minimal"
        else:
            return "normal"
    
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
    async def _chunk_document(documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks using a hierarchical chunking strategy.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        try:
            # Log input document info
            logger.info(f"Starting document chunking with {len(documents)} document(s)")
            
            if not documents:
                logger.error("No documents provided for chunking")
                return []
                
            # Log sample of first document
            first_doc = documents[0]
            sample_text = first_doc.page_content[:100] if hasattr(first_doc, 'page_content') else "No page_content"
            logger.info(f"First document sample: {sample_text}")
            
            # Hierarchical text splitter with overlapping windows
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Chunking completed. Generated {len(chunks)} chunks")
            
            # Log first chunk if available
            if chunks and len(chunks) > 0:
                sample_chunk = chunks[0].page_content[:100] if hasattr(chunks[0], 'page_content') else "No page_content"
                logger.info(f"First chunk sample: {sample_chunk}")
            
            return chunks
        except Exception as e:
            logger.error(f"Error during document chunking: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def _generate_embeddings(text: str) -> List[float]:
        """
        Generate embeddings for a text chunk.
        
        Args:
            text: Text chunk to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        # Validate and clean the text input
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input for embedding generation: {type(text)}")
            # Return a zero vector with the correct dimension as a fallback
            return [0.0] * EMBEDDING_DIM
            
        # Ensure text is a proper string and clean it
        try:
            # Clean text to remove problematic Unicode characters
            clean_text = DocumentService._clean_text(text.strip())
            
            if not clean_text:
                logger.warning("Empty text after cleaning, using fallback vector")
                return [0.0] * EMBEDDING_DIM
                
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: embedding_model.encode(clean_text).tolist())
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return a zero vector as fallback
            return [0.0] * EMBEDDING_DIM
            
    @staticmethod
    async def _generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks in a single batch.
        
        This is more efficient than generating embeddings one by one.
        
        Args:
            texts: List of text chunks to generate embeddings for
            
        Returns:
            List of embedding vectors for each text
        """
        if not texts:
            return []
            
        # Clean and validate all texts
        clean_texts = []
        indices = []
        
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text input at index {i}: {type(text)}")
                clean_texts.append("")  # Add placeholder for invalid text
            else:
                clean_text = DocumentService._clean_text(text.strip())
                clean_texts.append(clean_text)
            indices.append(i)
            
        # Generate embeddings for all valid texts in a single batch
        try:
            loop = asyncio.get_event_loop()
            # Filter out empty texts to avoid errors
            valid_texts = [t for t in clean_texts if t]
            valid_indices = [i for i, t in enumerate(clean_texts) if t]
            
            if not valid_texts:
                logger.warning("No valid texts for embedding generation")
                return [[0.0] * EMBEDDING_DIM for _ in texts]
                
            # Generate embeddings for valid texts
            embeddings = await loop.run_in_executor(
                None, 
                lambda: embedding_model.encode(valid_texts).tolist()
            )
            
            # Create result list with zero vectors for invalid texts
            result = [[0.0] * EMBEDDING_DIM for _ in texts]
            
            # Fill in embeddings for valid texts
            for i, embedding in zip(valid_indices, embeddings):
                result[i] = embedding
                
            return result
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * EMBEDDING_DIM for _ in texts]
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text by removing problematic Unicode characters.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove surrogate pairs (emoji characters often cause issues)
        cleaned_text = re.sub(r'[\ud800-\udfff]', '', text)
        
        # Replace other potentially problematic characters
        cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)  # Replace non-ASCII with spaces
        
        return cleaned_text
    
    @staticmethod
    async def get_document_by_id(
        db: AsyncSession,
        document_id: int,
        as_dict: bool = False
    ) -> Optional[Union[Document, Dict[str, Any]]]:
        """
        Get a document by ID.
        
        Args:
            db: Database session
            document_id: ID of the document to retrieve
            as_dict: Whether to return a dictionary instead of a SQLAlchemy model
            
        Returns:
            Document or None if not found
        """
        try:
            # Use a direct SQL query to avoid lazy loading issues
            result = await db.execute(
                text("""
                    SELECT 
                        id, 
                        user_id, 
                        filename, 
                        file_path, 
                        file_size, 
                        content_type, 
                        status,
                        page_count,
                        error_message,
                        created_at, 
                        updated_at
                    FROM documents
                    WHERE id = :document_id
                """),
                {"document_id": document_id}
            )
            
            # Important: result.first() is not awaitable, it's a synchronous method
            row = result.first()
            
            if not row:
                return None
            
            # Create Document object manually to avoid lazy loading
            document = Document(
                id=row.id,
                user_id=row.user_id,
                filename=row.filename,
                file_path=row.file_path,
                file_size=row.file_size,
                content_type=row.content_type,
                status=row.status,
                page_count=row.page_count,
                error_message=row.error_message,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            
            if as_dict:
                # Create dictionary directly rather than using utility function
                return {
                    "id": document.id,
                    "user_id": document.user_id,
                    "filename": document.filename,
                    "file_path": document.file_path,
                    "file_size": document.file_size,
                    "content_type": document.content_type,
                    "status": document.status,
                    "page_count": document.page_count,
                    "error_message": document.error_message,
                    "created_at": document.created_at,
                    "updated_at": document.updated_at,
                    "chunks": []  # Add empty chunks list to avoid issues with the schema
                }
            
            return document
        except Exception as e:
            logger.error(f"Error in get_document_by_id: {str(e)}")
            return None
    
    @staticmethod
    async def get_documents_by_user_id(
        db: AsyncSession, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 100,
        as_dict: bool = False
    ) -> Tuple[List[Union[Document, Dict[str, Any]]], int]:
        """
        Get all documents for a user.
        
        Args:
            db: Database session
            user_id: ID of the user
            skip: Number of records to skip
            limit: Maximum number of records to return
            as_dict: Whether to return dictionaries instead of SQLAlchemy models
            
        Returns:
            Tuple containing list of documents and total count
        """
        try:
            # Get total count using a direct SQL query
            count_result = await db.execute(
                text("SELECT COUNT(*) FROM documents WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            total = count_result.scalar() or 0
            
            # Get documents using a direct SQL query to avoid lazy loading
            # Include only the columns that actually exist in the database
            doc_query = text("""
                SELECT 
                    id, 
                    user_id, 
                    filename, 
                    file_path, 
                    file_size, 
                    content_type, 
                    status,
                    page_count,
                    error_message,
                    created_at, 
                    updated_at
                FROM documents
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :skip
            """)
            
            result = await db.execute(
                doc_query, 
                {"user_id": user_id, "limit": limit, "skip": skip}
            )
            
            documents = []
            for row in result:
                # Create Document objects manually to avoid lazy loading
                doc = Document(
                    id=row.id,
                    user_id=row.user_id,
                    filename=row.filename,
                    file_path=row.file_path,
                    file_size=row.file_size,
                    content_type=row.content_type,
                    status=row.status,
                    page_count=row.page_count,
                    error_message=row.error_message,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                )
                documents.append(doc)
            
            # Convert to dictionaries if requested
            if as_dict:
                # Create dictionaries directly
                document_dicts = []
                for doc in documents:
                    doc_dict = {
                        "id": doc.id,
                        "user_id": doc.user_id,
                        "filename": doc.filename,
                        "file_path": doc.file_path,
                        "file_size": doc.file_size,
                        "content_type": doc.content_type,
                        "status": doc.status,
                        "page_count": doc.page_count,
                        "error_message": doc.error_message,
                        "created_at": doc.created_at,
                        "updated_at": doc.updated_at,
                        # Add empty chunks list to avoid issues with the schema
                        "chunks": []
                    }
                    document_dicts.append(doc_dict)
                return document_dicts, total
            
            return documents, total
        except Exception as e:
            logger.error(f"Error in get_documents_by_user_id: {str(e)}")
            # Return empty list and zero count on error
            return [], 0
    
    @staticmethod
    async def search_documents(
        db: AsyncSession,
        user_id: int,
        query: str,
        limit: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for documents using semantic search.
        
        This function will:
        1. Generate an embedding vector for the query text
        2. Search for similar document chunks using vector similarity
        3. Return document chunks sorted by similarity score
        
        Args:
            db: Database session
            user_id: User ID to filter documents by
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of tuples containing document chunks and their similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = await DocumentService._generate_embeddings(query)
            logger.info(f"Generated embedding vector of length {len(query_embedding)}")
            
            try:
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
                await db.execute(text(setup_sql))
                await db.commit()
                
                # Now run the actual query
                logger.info("Executing vector search query")
                result = await db.execute(
                    text(vector_search_query), 
                    {
                        "user_id": user_id,
                        "embedding_json": f"[{','.join(str(x) for x in query_embedding)}]",
                        "limit": limit
                    }
                )
                
                chunks_with_scores = []
                for row in result:
                    # Create the DocumentChunk object without lazy loading attributes
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
                await db.rollback()
            
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
                    result = await db.execute(text(text_search_query), {"user_id": user_id, "limit": limit})
                    
                    chunks_with_scores = []
                    for row in result:
                        # Create the DocumentChunk object directly from row values to avoid lazy loading
                        chunk = DocumentChunk(
                            id=row.id,
                            document_id=row.document_id,
                            chunk_index=row.chunk_index,
                            text=row.text,
                            chunk_metadata=row.chunk_metadata,
                            created_at=row.created_at
                        )
                        
                        # Calculate a simple score based on keyword frequency
                        score = 0.5  # Default score for keyword matches
                        for keyword in keywords:
                            if len(keyword) > 3 and keyword.lower() in row.text.lower():
                                # Increase score based on number of occurrences
                                occurrences = row.text.lower().count(keyword.lower())
                                score += 0.1 * occurrences
                        
                        # Normalize score to be between 0 and 1
                        score = min(score, 0.95)
                        
                        chunks_with_scores.append((chunk, score))
                    
                    # Sort by score descending
                    chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    if chunks_with_scores:
                        logger.info(f"Keyword search found {len(chunks_with_scores)} results")
                        return chunks_with_scores
                
                # If no results from keyword search either
                logger.info("No results found from keyword search")
                return []
                
            except Exception as e:
                logger.error(f"Keyword search failed: {str(e)}")
                await db.rollback()
                return []
            
        except Exception as e:
            logger.error(f"Document search error: {str(e)}")
            return []

    @staticmethod
    async def delete_document(
        db: AsyncSession,
        document_id: int,
        user_id: int
    ) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            db: Database session
            document_id: ID of the document to delete
            user_id: ID of the user who owns the document
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Find the document
            query = select(Document).filter(Document.id == document_id)
            result = await db.execute(query)
            document = result.scalars().first()
            
            # Check if document exists and belongs to the user
            if not document:
                logger.error(f"Document not found: {document_id}")
                return False
                
            if document.user_id != user_id:
                logger.error(f"User {user_id} attempted to delete document {document_id} belonging to user {document.user_id}")
                return False
            
            # Delete associated document chunks first
            await db.execute(
                text("DELETE FROM document_chunks WHERE document_id = :document_id"),
                {"document_id": document_id}
            )
            
            # Delete the document
            await db.delete(document)
            await db.commit()
            
            logger.info(f"Document {document_id} successfully deleted by user {user_id}")
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    @staticmethod
    async def get_document_chunks(db: AsyncSession, document_id: int) -> List[DocumentChunk]:
        """
        Get all chunks for a document.
        
        Args:
            db: Database session
            document_id: ID of the document
            
        Returns:
            List of document chunks
        """
        # Fix: Use proper async query execution
        query = select(DocumentChunk).filter(DocumentChunk.document_id == document_id)
        result = await db.execute(query)
        chunks = result.scalars().all()
        return chunks 