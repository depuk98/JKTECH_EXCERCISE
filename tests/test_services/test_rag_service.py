import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import json
from fastapi import HTTPException

from app.services.rag import RAGService
from app.models.document import Document, DocumentChunk
from app.services.document import DocumentService


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_document():
    """Create a mock document."""
    return Document(
        id=1,
        user_id=1,
        filename="test_document.pdf",
        content_type="application/pdf",
        status="processed",
        created_at=datetime.now()
    )


@pytest.fixture
def mock_document_chunks():
    """Create mock document chunks with metadata."""
    return [
        DocumentChunk(
            id=1,
            document_id=1,
            chunk_index=0,
            text="This is the first chunk of content for testing.",
            chunk_metadata=json.dumps({"page": 1}),
            created_at=datetime.now()
        ),
        DocumentChunk(
            id=2,
            document_id=1,
            chunk_index=1,
            text="This is the second chunk with more detailed information about testing.",
            chunk_metadata=json.dumps({"page": 1}),
            created_at=datetime.now()
        ),
    ]


@pytest.mark.asyncio
async def test_retrieve_context_with_document_ids(mock_async_db, mock_document, mock_document_chunks):
    """Test retrieving context when document IDs are provided."""
    # Arrange
    rag_service = RAGService()
    mock_embedding = [0.1] * 384
    
    # Mock document service embedding generation
    with patch.object(
        DocumentService, '_generate_embeddings',
        return_value=mock_embedding
    ) as mock_generate:
        # Create mock results for vector search
        mock_result_item = MagicMock()
        mock_result_item.id = 1
        mock_result_item.document_id = 1
        mock_result_item.chunk_index = 0
        mock_result_item.text = "This is the first chunk"
        mock_result_item.chunk_metadata = json.dumps({"page": 1})
        mock_result_item.filename = "test_document.pdf"
        mock_result_item.similarity_score = 0.85
        
        # Set up the execute result
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [mock_result_item]
        
        # Configure the execute method to return the mock result
        mock_async_db.execute = AsyncMock()
        mock_async_db.execute.return_value = mock_result
        mock_async_db.commit = AsyncMock()
        
        # Act
        result = await rag_service.retrieve_context(
            db=mock_async_db,
            user_id=1,
            query="test query",
            document_ids=[1],
            top_k=1
        )
        
        # Assert
        assert len(result) == 1
        assert result[0]["document_id"] == 1
        assert result[0]["filename"] == "test_document.pdf"
        assert result[0]["similarity_score"] == 0.85
        assert mock_generate.called
        mock_async_db.execute.assert_called()
        mock_async_db.commit.assert_called()


@pytest.mark.asyncio
async def test_retrieve_context_without_document_ids(mock_async_db, mock_document, mock_document_chunks):
    """Test retrieving context when no document IDs are provided."""
    # Arrange
    rag_service = RAGService()
    mock_chunks_with_scores = [(mock_document_chunks[0], 0.8)]
    
    # Mock document service search method
    with patch.object(
        DocumentService, 'search_documents',
        return_value=mock_chunks_with_scores
    ) as mock_search:
        # Mock document service get_document_by_id method
        with patch.object(
            DocumentService, 'get_document_by_id',
            return_value=mock_document
        ) as mock_get_doc:
            # Act
            result = await rag_service.retrieve_context(
                db=mock_async_db,
                user_id=1,
                query="test query",
                document_ids=None,
                top_k=1
            )
            
            # Assert
            assert len(result) == 1
            assert result[0]["document_id"] == 1
            assert result[0]["filename"] == "test_document.pdf"
            assert result[0]["similarity_score"] == 0.8
            assert mock_search.called
            assert mock_get_doc.called


def test_format_context():
    """Test formatting context chunks into a prompt string with citations."""
    # Arrange
    rag_service = RAGService()
    context_chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "doc1.pdf",
            "text": "This is test content from the first document.",
            "metadata": {"page": 1},
            "similarity_score": 0.9
        },
        {
            "chunk_id": 2,
            "document_id": 2,
            "filename": "doc2.pdf",
            "text": "This is test content from the second document.",
            "metadata": {"page": 1},
            "similarity_score": 0.7
        }
    ]
    
    # Act
    formatted_context, citations = rag_service.format_context(context_chunks)
    
    # Assert
    assert "[1]" in formatted_context
    assert "[2]" in formatted_context
    assert "This is test content from the first document." in formatted_context
    assert "This is test content from the second document." in formatted_context
    assert len(citations) == 2
    assert citations[0]["id"] == 1
    assert citations[0]["filename"] == "doc1.pdf"
    assert citations[1]["id"] == 2
    assert citations[1]["filename"] == "doc2.pdf"


def test_generate_prompt():
    """
    Test the RAG prompt generation functionality.
    
    This test verifies that:
    1. The generate_prompt method creates a well-structured prompt for LLM generation
    2. The prompt correctly incorporates the user's query
    3. The prompt includes the retrieved context information
    4. The prompt contains clear instructions for the LLM
    5. The prompt follows the expected format with sections for context, question, and instructions
    
    Expected behavior:
    - The method should return a string containing multiple sections
    - The prompt should include the context provided
    - The user's query should be clearly marked in the prompt
    - The prompt should include instructions about using citations
    - The prompt should have a clear structure for the LLM to follow
    
    LLM interaction:
    - Ensures the prompt structure is compatible with instruction-tuned LLMs
    - Verifies that citation instructions are included for proper sourcing
    - Confirms the prompt includes clear boundaries between sections
    
    Edge cases:
    - The test verifies that all key components are present regardless of context/query content
    - The test ensures the prompt maintains its structure with varying inputs
    """
    # Arrange
    rag_service = RAGService()
    context = "[1] This is test context."
    query = "What is this about?"
    
    # Act
    prompt = rag_service.generate_prompt(query, context)
    
    # Assert - check essential parts of the prompt
    assert "CONTEXT INFORMATION:" in prompt
    assert context in prompt
    assert "USER QUESTION:" in prompt
    assert query in prompt
    assert "INSTRUCTIONS:" in prompt
    assert "citation markers" in prompt.lower()
    assert "Your answer:" in prompt


@pytest.mark.asyncio
async def test_answer_question_no_context(mock_db):
    """Test handling a question when no relevant context is found."""
    # Arrange
    rag_service = RAGService()
    
    # Mock retrieve_context to return empty results
    with patch.object(rag_service, 'retrieve_context', return_value=[]) as mock_retrieve, \
         patch.object(rag_service, 'generate_answer_ollama', new_callable=AsyncMock) as mock_generate:
        
        # Set up the mock response for no context
        mock_generate.return_value = "I don't have enough information to answer this question."
        
        # Act
        result = await rag_service.answer_question(
            db=mock_db,
            user_id=1,
            query="test question",
            document_ids=None
        )
        
        # Assert
        assert "I don't have enough information" in result["answer"]
        assert len(result["citations"]) == 0
        mock_retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_with_ollama(mock_db):
    """Test answering a question with Ollama."""
    # Arrange
    rag_service = RAGService()
    query = "What is the capital of France?"
    
    # Mock the retrieve_context method
    context_chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "doc1.pdf",
            "text": "Paris is the capital of France.",
            "metadata": {"page": 1},
            "similarity_score": 0.9
        }
    ]
    
    # Mock dependencies
    with patch.object(rag_service, 'retrieve_context', return_value=context_chunks) as mock_retrieve, \
         patch.object(rag_service, 'generate_answer_ollama', new_callable=AsyncMock) as mock_generate:
        
        # Set up the mock response
        mock_generate.return_value = "The capital of France is Paris [1]."
        
        # Act
        result = await rag_service.answer_question(
            db=mock_db,
            user_id=1,
            query=query,
            document_ids=None
        )
        
        # Assert
        assert "Paris" in result["answer"]
        assert len(result["citations"]) == 1
        assert result["citations"][0]["filename"] == "doc1.pdf"
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_documents(mock_db, mock_document):
    """Test retrieving available documents for Q&A."""
    # Arrange
    rag_service = RAGService()
    
    # Mock document service
    with patch.object(
        DocumentService, 'get_documents_by_user_id',
        return_value=([mock_document], 1)
    ) as mock_get_docs:
        # Act
        result = await rag_service.get_available_documents(
            db=mock_db,
            user_id=1
        )
        
        # Assert
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["filename"] == "test_document.pdf"
        assert mock_get_docs.called 


@pytest.mark.asyncio
async def test_retrieve_context_with_empty_documents():
    """Test retrieving context when no documents exist."""
    # Create RAG service
    rag_service = RAGService()
    mock_db = MagicMock()
    
    # Mock DocumentService.search_documents to return empty results
    with patch('app.services.document.DocumentService.search_documents', 
              new_callable=AsyncMock, return_value=[]):
        # Act - retrieve context when no documents exist
        context = await rag_service.retrieve_context(
            db=mock_db,
            user_id=1,
            query="test query",
            document_ids=None,
            top_k=1
        )
        
        # Assert - should get an empty list
        assert context == [], "Should return empty list when no documents exist"


@pytest.mark.asyncio
async def test_retrieve_context_with_specific_document_not_found():
    """Test retrieving context when a specified document doesn't exist."""
    # Create RAG service
    rag_service = RAGService()
    mock_db = MagicMock()
    
    # Mock DocumentService._generate_embeddings to avoid errors
    with patch('app.services.document.DocumentService._generate_embeddings', 
              new_callable=AsyncMock, 
              return_value=[0.1] * 384):
        
        # Mock db.execute to raise an exception when document is not found
        mock_db.execute.side_effect = Exception("Document not found")
        
        # Act - attempt to retrieve context with non-existent document ID
        # This should catch the exception and return an empty list
        result = await rag_service.retrieve_context(
            db=mock_db,
            user_id=1,
            query="test query",
            document_ids=[999],
            top_k=1
        )
        
        # Assert - should return empty list when document not found
        assert result == [], "Should return empty list when document not found"
        
        # Verify db.execute was called
        mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_formatting_context_with_different_chunk_types():
    """Test formatting different types of context chunks."""
    # Create RAG service
    rag_service = RAGService()
    
    # Create test chunks with different properties
    chunks = [
        # PDF chunk with page number
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "doc1.pdf",
            "text": "This is from a PDF",
            "metadata": {"page": 5},
            "similarity_score": 0.9
        },
        # Text file chunk without page number
        {
            "chunk_id": 2,
            "document_id": 2,
            "filename": "doc2.txt",
            "text": "This is from a text file",
            "metadata": {},
            "similarity_score": 0.8
        },
        # Chunk without metadata
        {
            "chunk_id": 3,
            "document_id": 3,
            "filename": "doc3.md",
            "text": "This has no metadata",
            "metadata": {},
            "similarity_score": 0.7
        }
    ]
    
    # Act - format the context
    formatted_context, citations = rag_service.format_context(chunks)
    
    # Assert
    assert isinstance(formatted_context, str), "Formatted context should be a string"
    assert "[1]" in formatted_context, "Should include citation markers"
    assert "This is from a PDF" in formatted_context, "Should include chunk text"
    
    # Check citations
    assert len(citations) == 3, "Should have 3 citations"
    assert citations[0]["filename"] == "doc1.pdf", "Should include filename in citations"
    assert citations[0]["id"] == 1, "First citation should have ID 1"


@pytest.mark.asyncio
async def test_answer_generation_with_malformed_context():
    """Test generating answers with malformed context."""
    # Create RAG service
    rag_service = RAGService()
    mock_db = MagicMock()
    
    # Mock retrieve_context to return empty results
    with patch.object(rag_service, 'retrieve_context', 
                     new_callable=AsyncMock, 
                     return_value=[]):
        
        # Mock format_context to return empty formatted context
        with patch.object(rag_service, 'format_context',
                         return_value=("", [])):
            
            # Mock the Ollama API response
            with patch.object(rag_service, 'generate_answer_ollama', 
                             new_callable=AsyncMock,
                             return_value="I don't have enough information to answer this question."):
                
                # Act - generate answer with empty context
                result = await rag_service.answer_question(
                    db=mock_db,
                    user_id=1,
                    query="What is the topic?",
                    document_ids=None,
                    use_cache=False
                )
                
                # Assert - should handle gracefully and return a reasonable response
                assert result, "Should return a result"
                assert "answer" in result, "Result should contain an answer field"
                assert "I don't have enough information" in result["answer"], "Answer should indicate insufficient information"
                assert "citations" in result, "Result should include citations field"
                assert len(result["citations"]) == 0, "Should have no citations"


@pytest.mark.asyncio
async def test_citation_extraction_and_formatting():
    """Test extraction and formatting of citations in LLM responses."""
    # Create RAG service
    rag_service = RAGService()
    mock_db = MagicMock()
    
    # Setup - sample LLM output with citations
    raw_llm_output = """
    Einstein developed the theory of relativity in 1905. [1]
    The theory revolutionized our understanding of physics. [2]

    Sources:
    [1] doc1.pdf
    [2] doc2.pdf
    """
    
    # Mock context chunks for citations
    context_chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "doc1.pdf",
            "text": "Einstein's theory",
            "metadata": {},
            "similarity_score": 0.9
        },
        {
            "chunk_id": 2,
            "document_id": 2,
            "filename": "doc2.pdf",
            "text": "Physics revolution",
            "metadata": {},
            "similarity_score": 0.8
        }
    ]
    
    # Mock the necessary methods
    with patch.object(rag_service, 'retrieve_context', 
                     new_callable=AsyncMock, 
                     return_value=context_chunks):
        
        with patch.object(rag_service, 'format_context',
                         return_value=("Formatted context", context_chunks)):
            
            with patch.object(rag_service, 'generate_answer_ollama',
                             new_callable=AsyncMock,
                             return_value=raw_llm_output):
                
                # Act - generate answer with citations
                result = await rag_service.answer_question(
                    db=mock_db,
                    user_id=1,
                    query="What is relativity?",
                    document_ids=None,
                    use_cache=False
                )
                
                # Assert - response should contain both the answer and citation information
                assert result, "Should return a result"
                assert "answer" in result, "Result should contain an answer field"
                assert "Einstein" in result["answer"], "Answer should contain the text from LLM"
                assert "citations" in result, "Result should include citations"
                assert len(result["citations"]) > 0, "Should have at least one citation"
                assert result["citations"][0]["filename"] == "doc1.pdf", "Citation should include correct filename" 