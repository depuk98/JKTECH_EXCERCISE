import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.models.document import Document
from app.services.rag import RAGService


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


@pytest.mark.asyncio
async def test_get_available_documents(client, mock_token_headers, mock_document):
    """Test the endpoint to get available documents for Q&A."""
    # Arrange
    available_docs = [{
        "id": 1,
        "filename": "test_document.pdf",
        "status": "processed",
        "created_at": mock_document.created_at.isoformat()
    }]
    
    # Mock the RAG service
    with patch.object(
        RAGService, 'get_available_documents',
        return_value=available_docs
    ) as mock_get_docs:
        # Act - Use the main documents endpoint instead of the duplicate endpoint
        response = await client.get("/api/documents", headers=mock_token_headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # The documents API returns a different structure compared to the QA API
        # It returns an object with a 'documents' array and 'total' count
        assert "documents" in data
        assert len(data["documents"]) >= 1
        
        # From here, we would filter processed documents in the frontend
        processed_docs = [doc for doc in data["documents"] if doc["status"] == "processed"]
        # Need at least one processed document to continue with tests
        assert len(processed_docs) >= 1


@pytest.mark.asyncio
async def test_ask_question_success(client, mock_token_headers):
    """Test asking a question successfully."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    answer_result = {
        "answer": "The main topic is artificial intelligence.",
        "citations": [{
            "id": 1,
            "filename": "test_document.pdf",
            "page": 1
        }],
        "metadata": {
            "model_used": "gpt-3.5-turbo",
            "query_time_ms": 500
        }
    }
    
    # Mock the RAG service
    with patch.object(
        RAGService, 'answer_question',
        return_value=answer_result
    ) as mock_answer:
        # Act
        response = await client.post(
            "/api/qa/ask",
            json=question_data,
            headers=mock_token_headers
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The main topic is artificial intelligence."
        assert len(data["citations"]) == 1
        assert data["citations"][0]["filename"] == "test_document.pdf"
        assert mock_answer.called


@pytest.mark.asyncio
async def test_ask_question_with_all_documents(client, mock_token_headers):
    """Test asking a question across all documents."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": None  # No specific documents, use all
    }
    
    answer_result = {
        "answer": "The main topic is artificial intelligence.",
        "citations": [{
            "id": 1,
            "filename": "test_document.pdf",
            "page": 1
        }],
        "metadata": {
            "model_used": "gpt-3.5-turbo",
            "query_time_ms": 500
        }
    }
    
    # Mock the RAG service
    with patch.object(
        RAGService, 'answer_question',
        return_value=answer_result
    ) as mock_answer:
        # Act
        response = await client.post(
            "/api/qa/ask",
            json=question_data,
            headers=mock_token_headers
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The main topic is artificial intelligence."
        assert mock_answer.called
        # Check that the document_ids parameter was passed as None
        mock_answer.assert_called_with(
            db=mock_answer.call_args[1]['db'],
            user_id=mock_answer.call_args[1]['user_id'],
            query="What is the main topic?",
            document_ids=None
        )


@pytest.mark.asyncio
async def test_ask_question_invalid_query(client, mock_token_headers):
    """Test asking a question with an invalid or empty query."""
    # Arrange
    question_data = {
        "query": "",  # Empty query
        "document_ids": [1]
    }
    
    # Act
    response = await client.post(
            "/api/qa/ask",
            json=question_data,
            headers=mock_token_headers
        )
    
    # Assert
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ask_question_error_handling(client, mock_token_headers):
    """Test error handling when answering a question."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    # Mock the RAG service to raise an exception
    with patch.object(
        RAGService, 'answer_question',
        side_effect=Exception("Test error")
    ) as mock_answer:
        # Act
        response = await client.post(
            "/api/qa/ask",
            json=question_data,
            headers=mock_token_headers
        )
        
        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Failed to process question" in data["error"]
        assert mock_answer.called 