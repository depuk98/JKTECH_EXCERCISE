import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import json

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
async def test_retrieve_context_with_document_ids(mock_db, mock_document, mock_document_chunks):
    """Test retrieving context when document IDs are provided."""
    # Arrange
    rag_service = RAGService()
    mock_embedding = [0.1] * 384
    
    # Mock document service embedding generation
    with patch.object(
        DocumentService, '_generate_embeddings',
        return_value=mock_embedding
    ) as mock_generate:
        # Setup the mock db.execute response for the first query (pgvector function setup)
        mock_db.execute = MagicMock()
        mock_execute = mock_db.execute
        
        # Setup the mock db.execute response for the second query (vector search)
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [
            MagicMock(
                id=1,
                document_id=1,
                chunk_index=0,
                text="This is the first chunk",
                chunk_metadata={"page": 1},
                filename="test_document.pdf",
                similarity_score=0.85
            )
        ]
        mock_execute.return_value = mock_result
        
        # Act
        result = await rag_service.retrieve_context(
            db=mock_db,
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
        mock_db.execute.assert_called()
        mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_retrieve_context_without_document_ids(mock_db, mock_document, mock_document_chunks):
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
                db=mock_db,
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
    """Test generating a prompt with context and query."""
    # Arrange
    rag_service = RAGService()
    context = "[1] This is test context."
    query = "What is this about?"
    
    # Act
    prompt = rag_service.generate_prompt(query, context)
    
    # Assert
    assert "Context:" in prompt
    assert "[1] This is test context." in prompt
    assert "Question: What is this about?" in prompt
    

@pytest.mark.asyncio
async def test_answer_question_no_context(mock_db):
    """Test handling a question when no relevant context is found."""
    # Arrange
    rag_service = RAGService()
    
    # Mock retrieve_context to return empty results
    with patch.object(
        rag_service, 'retrieve_context',
        return_value=[]
    ) as mock_retrieve:
        # Act
        result = await rag_service.answer_question(
            db=mock_db,
            user_id=1,
            query="test question",
            document_ids=None
        )
        
        # Assert
        assert "I couldn't find any relevant information" in result["answer"]
        assert len(result["citations"]) == 0
        assert "error" in result["metadata"]
        assert result["metadata"]["error"] == "No relevant context found"
        assert mock_retrieve.called


@pytest.mark.asyncio
async def test_answer_question_with_openai(mock_db):
    """Test answering a question using the OpenAI API."""
    # Arrange
    rag_service = RAGService()
    rag_service.use_openai = True
    
    context_chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "doc1.pdf",
            "text": "This is test content.",
            "metadata": {"page": 1},
            "similarity_score": 0.9
        }
    ]
    
    # Mock retrieve_context
    with patch.object(
        rag_service, 'retrieve_context',
        return_value=context_chunks
    ) as mock_retrieve:
        # Mock generate_answer_openai
        with patch.object(
            rag_service, 'generate_answer_openai',
            return_value="This is a test answer based on the provided context."
        ) as mock_generate:
            # Act
            result = await rag_service.answer_question(
                db=mock_db,
                user_id=1,
                query="test question",
                document_ids=None
            )
            
            # Assert
            assert result["answer"] == "This is a test answer based on the provided context."
            assert len(result["citations"]) == 1
            assert result["citations"][0]["filename"] == "doc1.pdf"
            assert mock_retrieve.called
            assert mock_generate.called


@pytest.mark.asyncio
async def test_answer_question_with_ollama(mock_db):
    """Test answering a question using the Ollama API."""
    # Arrange
    rag_service = RAGService()
    rag_service.use_openai = False
    
    context_chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "doc1.pdf",
            "text": "This is test content.",
            "metadata": {"page": 1},
            "similarity_score": 0.9
        }
    ]
    
    # Mock retrieve_context
    with patch.object(
        rag_service, 'retrieve_context',
        return_value=context_chunks
    ) as mock_retrieve:
        # Mock generate_answer_ollama
        with patch.object(
            rag_service, 'generate_answer_ollama',
            return_value="This is a test answer from Ollama based on the provided context."
        ) as mock_generate:
            # Act
            result = await rag_service.answer_question(
                db=mock_db,
                user_id=1,
                query="test question",
                document_ids=None
            )
            
            # Assert
            assert result["answer"] == "This is a test answer from Ollama based on the provided context."
            assert len(result["citations"]) == 1
            assert result["citations"][0]["filename"] == "doc1.pdf"
            assert mock_retrieve.called
            assert mock_generate.called


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