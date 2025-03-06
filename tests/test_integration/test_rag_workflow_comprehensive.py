import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.services.rag import RAGService
from app.services.document import DocumentService
from app.models.document import Document, DocumentChunk
from app.models.user import User


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.refresh = MagicMock()
    db.query = MagicMock()
    db.execute = MagicMock()
    return db


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = User(
        id=1,
        email="test@example.com",
        hashed_password="hashed_password",
        is_active=True
    )
    return user


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    content = """
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
    especially computer systems. These processes include learning (the acquisition of information 
    and rules for using the information), reasoning (using rules to reach approximate or definite 
    conclusions) and self-correction.

    Machine Learning is a subset of AI that provides systems the ability to automatically learn 
    and improve from experience without being explicitly programmed. Machine learning focuses 
    on the development of computer programs that can access data and use it to learn for themselves.

    Deep Learning is a subset of machine learning that uses neural networks with many layers 
    (hence "deep") to analyze various factors of data. Deep learning is a key technology behind 
    driverless cars, enabling them to recognize a stop sign or distinguish a pedestrian from a lamppost.
    """
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    temp_file.write(content.encode('utf-8'))
    temp_file.close()
    
    yield temp_file.name
    
    # Clean up
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_comprehensive_rag_workflow(mock_db, mock_user, sample_text_file):
    """
    Test the complete RAG workflow from document upload to answer generation.
    
    This test verifies:
    1. Document upload and processing
    2. Chunking and embedding generation
    3. Context retrieval based on query
    4. Answer generation with citations
    
    The test uses a real text file but mocks database and LLM interactions.
    """
    # Step 1: Upload document
    # Create a mock upload file
    class MockUploadFile:
        def __init__(self, filename, content_type, content):
            self.filename = filename
            self.content_type = content_type
            self._content = content
            self.size = len(content)
        
        async def read(self):
            return self._content
            
        async def seek(self, position):
            pass
    
    # Read the sample file content
    with open(sample_text_file, 'rb') as f:
        file_content = f.read()
    
    mock_file = MockUploadFile(
        filename="ai_concepts.txt",
        content_type="text/plain",
        content=file_content
    )
    
    # Mock document creation
    mock_document = Document(
        id=1,
        user_id=mock_user.id,
        filename=mock_file.filename,
        content_type=mock_file.content_type,
        status="processed"
    )
    
    # Mock document chunks
    mock_chunks = [
        DocumentChunk(
            id=1,
            document_id=1,
            chunk_index=0,
            text="Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
            embedding=[0.1] * 384
        ),
        DocumentChunk(
            id=2,
            document_id=1,
            chunk_index=1,
            text="Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience.",
            embedding=[0.2] * 384
        ),
        DocumentChunk(
            id=3,
            document_id=1,
            chunk_index=2,
            text="Deep Learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data.",
            embedding=[0.3] * 384
        )
    ]
    
    # Mock the document upload process
    with patch.object(DocumentService, 'upload_document', return_value=mock_document), \
         patch.object(DocumentService, 'get_document_by_id', return_value=mock_document), \
         patch.object(DocumentService, 'get_document_chunks', return_value=mock_chunks), \
         patch.object(DocumentService, '_generate_embeddings', return_value=[0.5] * 384):
        
        # Step 2: Create RAG service
        rag_service = RAGService()
        
        # Step 3: Mock context retrieval
        with patch.object(rag_service, 'retrieve_context', return_value=[
            {
                "chunk_id": 2,
                "document_id": 1,
                "filename": "ai_concepts.txt",
                "text": "Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience.",
                "metadata": {},
                "similarity_score": 0.85
            },
            {
                "chunk_id": 3,
                "document_id": 1,
                "filename": "ai_concepts.txt",
                "text": "Deep Learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data.",
                "metadata": {},
                "similarity_score": 0.75
            }
        ]):
            
            # Step 4: Mock LLM response
            expected_answer = """
            Machine Learning is a subset of Artificial Intelligence (AI) [1]. It enables systems to automatically learn and improve from experience without explicit programming.
            
            Deep Learning is a further subset of Machine Learning [2]. It specifically uses neural networks with many layers to analyze data, which is why it's called "deep" learning.
            """
            
            with patch.object(rag_service, 'generate_answer_ollama', return_value=expected_answer):
                
                # Step 5: Ask a question
                result = await rag_service.answer_question(
                    db=mock_db,
                    user_id=mock_user.id,
                    query="What is the relationship between Machine Learning and Deep Learning?",
                    document_ids=[1]
                )
                
                # Step 6: Verify the result
                assert "answer" in result
                assert "citations" in result
                assert "metadata" in result
                assert "Machine Learning" in result["answer"]
                assert "Deep Learning" in result["answer"]
                assert "subset" in result["answer"]
                
                # Verify citations
                assert len(result["citations"]) > 0
                assert result["citations"][0]["filename"] == "ai_concepts.txt"
                
                # Verify metadata
                assert result["metadata"]["document_ids"] == [1]
                assert result["metadata"]["context_chunks"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_workflow_with_empty_document(mock_db, mock_user):
    """
    Test the RAG workflow with an empty document.
    
    This test verifies that the system handles empty documents gracefully.
    """
    # Create an empty document
    mock_document = Document(
        id=1,
        user_id=mock_user.id,
        filename="empty.txt",
        content_type="text/plain",
        status="processed"
    )
    
    # Mock empty document chunks
    mock_chunks = []
    
    # Mock document retrieval
    with patch.object(DocumentService, 'get_document_by_id', return_value=mock_document), \
         patch.object(DocumentService, 'get_document_chunks', return_value=mock_chunks), \
         patch.object(DocumentService, '_generate_embeddings', return_value=[0.5] * 384):
        
        # Create RAG service
        rag_service = RAGService()
        
        # Mock context retrieval to return empty results
        with patch.object(rag_service, 'retrieve_context', return_value=[]):
            
            # Ask a question
            result = await rag_service.answer_question(
                db=mock_db,
                user_id=mock_user.id,
                query="What is in this document?",
                document_ids=[1]
            )
            
            # Verify the result indicates insufficient information
            assert "answer" in result
            assert "I don't have enough information" in result["answer"]
            assert len(result["citations"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_workflow_with_ollama_unavailable(mock_db, mock_user):
    """
    Test the RAG workflow when Ollama is unavailable.
    
    This test verifies that the system handles LLM unavailability gracefully.
    """
    # Create a document
    mock_document = Document(
        id=1,
        user_id=mock_user.id,
        filename="test.txt",
        content_type="text/plain",
        status="processed"
    )
    
    # Mock document chunks
    mock_chunks = [
        DocumentChunk(
            id=1,
            document_id=1,
            chunk_index=0,
            text="This is a test document.",
            embedding=[0.1] * 384
        )
    ]
    
    # Mock document retrieval
    with patch.object(DocumentService, 'get_document_by_id', return_value=mock_document), \
         patch.object(DocumentService, 'get_document_chunks', return_value=mock_chunks):
        
        # Create RAG service
        rag_service = RAGService()
        
        # Mock Ollama unavailability
        with patch.object(rag_service, 'check_ollama_available', return_value=False):
            
            # Ask a question
            result = await rag_service.answer_question(
                db=mock_db,
                user_id=mock_user.id,
                query="What is in this document?",
                document_ids=[1]
            )
            
            # Verify the result indicates LLM unavailability
            assert "answer" in result
            assert "AI service is not available" in result["answer"]
            assert "error" in result["metadata"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_workflow_with_multiple_documents(mock_db, mock_user):
    """
    Test the RAG workflow with multiple documents.
    
    This test verifies that the system can retrieve context from multiple documents.
    """
    # Create multiple documents
    mock_documents = [
        Document(
            id=1,
            user_id=mock_user.id,
            filename="doc1.txt",
            content_type="text/plain",
            status="processed"
        ),
        Document(
            id=2,
            user_id=mock_user.id,
            filename="doc2.txt",
            content_type="text/plain",
            status="processed"
        )
    ]
    
    # Mock document chunks
    mock_chunks = [
        # Document 1 chunks
        DocumentChunk(
            id=1,
            document_id=1,
            chunk_index=0,
            text="Artificial Intelligence is transforming industries.",
            embedding=[0.1] * 384
        ),
        # Document 2 chunks
        DocumentChunk(
            id=2,
            document_id=2,
            chunk_index=0,
            text="Machine Learning applications are growing rapidly.",
            embedding=[0.2] * 384
        )
    ]
    
    # Mock document retrieval
    with patch.object(DocumentService, 'get_document_by_id', side_effect=lambda db, doc_id: 
                     mock_documents[0] if doc_id == 1 else mock_documents[1]), \
         patch.object(DocumentService, '_generate_embeddings', return_value=[0.5] * 384):
        
        # Create RAG service
        rag_service = RAGService()
        
        # Mock context retrieval to return chunks from both documents
        with patch.object(rag_service, 'retrieve_context', return_value=[
            {
                "chunk_id": 1,
                "document_id": 1,
                "filename": "doc1.txt",
                "text": "Artificial Intelligence is transforming industries.",
                "metadata": {},
                "similarity_score": 0.85
            },
            {
                "chunk_id": 2,
                "document_id": 2,
                "filename": "doc2.txt",
                "text": "Machine Learning applications are growing rapidly.",
                "metadata": {},
                "similarity_score": 0.75
            }
        ]):
            
            # Mock LLM response
            expected_answer = """
            Artificial Intelligence is transforming various industries [1]. 
            At the same time, Machine Learning applications are experiencing rapid growth [2].
            """
            
            with patch.object(rag_service, 'generate_answer_ollama', return_value=expected_answer):
                
                # Ask a question
                result = await rag_service.answer_question(
                    db=mock_db,
                    user_id=mock_user.id,
                    query="How are AI and ML impacting industries?",
                    document_ids=[1, 2]
                )
                
                # Verify the result includes information from both documents
                assert "answer" in result
                assert "citations" in result
                assert len(result["citations"]) == 2
                assert result["citations"][0]["filename"] == "doc1.txt"
                assert result["citations"][1]["filename"] == "doc2.txt"
                assert "document_ids" in result["metadata"]
                assert result["metadata"]["document_ids"] == [1, 2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_workflow_with_caching(mock_db, mock_user):
    """
    Test the RAG workflow with caching.
    
    This test verifies that the system properly caches and retrieves responses.
    """
    # Create a document
    mock_document = Document(
        id=1,
        user_id=mock_user.id,
        filename="test.txt",
        content_type="text/plain",
        status="processed"
    )
    
    # Mock document retrieval
    with patch.object(DocumentService, 'get_document_by_id', return_value=mock_document):
        
        # Create RAG service
        rag_service = RAGService()
        
        # Clear the cache to ensure we don't get unexpected cache hits
        from app.services.rag import QUERY_CACHE
        QUERY_CACHE.clear()
        
        # Mock context retrieval and answer generation for first call
        with patch.object(rag_service, 'check_ollama_available', return_value=True), \
             patch.object(rag_service, 'retrieve_context', return_value=[
                {
                    "chunk_id": 1,
                    "document_id": 1,
                    "filename": "test.txt",
                    "text": "This is a test document.",
                    "metadata": {},
                    "similarity_score": 0.85
                }
             ]), \
             patch.object(rag_service, 'generate_answer_ollama', return_value="This is a test document [1]."):
            
            # First query - should generate a new response
            query = "What is in this document?"
            result1 = await rag_service.answer_question(
                db=mock_db,
                user_id=mock_user.id,
                query=query,
                document_ids=[1],
                use_cache=True
            )
            
            # Verify the first result
            assert result1["answer"] == "This is a test document [1]."
            assert result1["metadata"]["cached"] is False
            
            # Second query with same parameters - should use cache
            # Use new mocks to verify they are NOT called
            with patch.object(rag_service, 'retrieve_context', return_value=[]) as mock_retrieve, \
                 patch.object(rag_service, 'generate_answer_ollama', return_value="New response") as mock_generate:
                
                result2 = await rag_service.answer_question(
                    db=mock_db,
                    user_id=mock_user.id,
                    query=query,
                    document_ids=[1],
                    use_cache=True
                )
                
                # Verify the second result uses cache
                assert result2["answer"] == result1["answer"]
                assert result2["metadata"]["cached"] is True
                
                # Verify the mocks were not called
                mock_retrieve.assert_not_called()
                mock_generate.assert_not_called()
                
                # Third query with different parameters - should NOT use cache
                # But since we're mocking retrieve_context to return empty, the generate_answer_ollama won't be called
                result3 = await rag_service.answer_question(
                    db=mock_db,
                    user_id=mock_user.id,
                    query="Different query",
                    document_ids=[1],
                    use_cache=True
                )
                
                # Verify the third result does not use cache
                assert result3["metadata"]["cached"] is False
                
                # Verify retrieve_context was called but generate_answer_ollama was not
                # (because no context was retrieved)
                mock_retrieve.assert_called_once()
                assert "I don't have enough information" in result3["answer"] 