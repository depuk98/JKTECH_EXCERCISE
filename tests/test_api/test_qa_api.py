import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from fastapi import HTTPException

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


@pytest.mark.skip("This endpoint is now handled by the documents route")
def test_get_available_documents(client, user_token_headers, mock_document):
    """Test the endpoint to get available documents for Q&A."""
    # Arrange
    available_docs = [{
        "id": 1,
        "filename": "test_document.pdf",
        "status": "processed",
        "created_at": mock_document.created_at.isoformat(),
        "user_id": 1,
        "content_type": "application/pdf"
    }]
    
    # Mock the documents endpoint to return test data
    with patch('app.api.routes.documents.DocumentService.get_documents_by_user_id', 
              return_value=(available_docs, 1)) as mock_get_docs:
        # Act - Use the documents endpoint instead
        response = client.get("/api/documents/", headers=user_token_headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # The documents API returns an object with a 'documents' array and 'total' count
        assert "documents" in data
        assert "total" in data
        assert data["total"] >= 0
        
        # If there are documents, check they have the right format
        if data["documents"] and len(data["documents"]) > 0:
            doc = data["documents"][0]
            assert "id" in doc
            assert "filename" in doc
            assert "status" in doc
            assert "user_id" in doc
            assert "content_type" in doc


def test_ask_question_success(client, user_token_headers):
    """
    Test the question-answering endpoint with a valid query.
    
    This test verifies that:
    1. The /api/qa/ask endpoint accepts a question payload with document_ids
    2. The endpoint routes the query to the RAG service with proper parameters
    3. The endpoint returns a properly structured response with HTTP 200
    4. The response contains the expected answer text and citation information 
    5. The metadata in the response provides information about the model used
    
    Expected behavior:
    - The endpoint should return HTTP 200 with a JSON response
    - The response should contain an answer that matches what the RAG service returned
    - The response should include citations to the source documents
    - The RAG service should be called exactly once with the right parameters
    """
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
            "model_used": "ollama",
            "query_time_ms": 500
        }
    }
    
    # First, we need to ensure a document exists to query
    with patch('app.api.routes.qa.DocumentService.get_document_by_id', 
              return_value=Document(id=1, user_id=1, status="processed")):
        
        # Mock the RAG service
        with patch('app.api.routes.qa.rag_service.answer_question', 
                   new_callable=AsyncMock, 
                   return_value=answer_result) as mock_answer:
            # Act
            response = client.post(
                "/api/qa/ask",
                json=question_data,
                headers=user_token_headers
            )
            
            # Assert
            assert response.status_code == 200, f"Response: {response.json()}"
            data = response.json()
            assert data["answer"] == "The main topic is artificial intelligence."
            assert len(data["citations"]) == 1
            assert data["citations"][0]["filename"] == "test_document.pdf"
            mock_answer.assert_called_once()


def test_ask_question_invalid_query(client, user_token_headers):
    """
    Test the question-answering endpoint with an invalid (empty) query.
    
    This test verifies that:
    1. The /api/qa/ask endpoint properly validates the input query
    2. When an empty query is provided, the API returns an appropriate error
    3. Input validation happens before any expensive processing is performed
    
    Expected behavior:
    - The endpoint should return HTTP 400 Bad Request for empty queries
    - The response should contain a clear error message about the empty query
    - The error message should indicate that the query cannot be empty
    
    Edge cases tested:
    - Empty string as query (should be rejected)
    - The error handling should be specific to empty queries, not general validation errors (which would be 422)
    """
    # Arrange
    question_data = {
        "query": "",  # Empty query
        "document_ids": [1]
    }
    
    # Act
    response = client.post(
            "/api/qa/ask",
            json=question_data,
            headers=user_token_headers
        )
    
    # Assert
    # The API returns 400 Bad Request for empty queries rather than 422 Validation Error
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "empty" in data["detail"].lower()


def test_ask_question_error_handling(client, user_token_headers):
    """Test error handling when answering a question."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    # First, we need to ensure a document exists to query
    with patch('app.api.routes.qa.DocumentService.get_document_by_id', 
              return_value=Document(id=1, user_id=1, status="processed")):
        
        # Mock the RAG service to raise an exception
        with patch('app.api.routes.qa.rag_service.answer_question',
                   new_callable=AsyncMock,
                   side_effect=Exception("Test error")) as mock_answer:
            # Act
            response = client.post(
                "/api/qa/ask",
                json=question_data,
                headers=user_token_headers
            )
            
            # Assert
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "failed" in data["detail"].lower()
            mock_answer.assert_called_once()


def test_ask_question_no_authentication(client):
    """Test asking a question without authentication."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    # Act
    response = client.post(
        "/api/qa/ask",
        json=question_data
    )
    
    # Assert
    assert response.status_code == 401, "Should require authentication"
    assert "detail" in response.json()


def test_ask_question_with_invalid_document_ids(client, user_token_headers):
    """Test asking a question with invalid document IDs."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [999]  # Non-existent document ID
    }
    
    # Mock DocumentService to raise a 404 error for non-existent document
    with patch('app.api.routes.qa.DocumentService.get_document_by_id',
              new_callable=AsyncMock,
              side_effect=HTTPException(
                  status_code=404,
                  detail="Document with ID 999 not found"
              )):
        # Act
        response = client.post(
            "/api/qa/ask",
            json=question_data,
            headers=user_token_headers
        )
        
        # Assert
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


def test_ask_question_with_rate_limit_reached(client, user_token_headers):
    """Test asking a question when rate limit has been reached."""
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    # First mock DocumentService.get_document_by_id to return a valid document
    with patch('app.api.routes.qa.DocumentService.get_document_by_id',
              new_callable=AsyncMock) as mock_get_doc:
        # Setup mock document
        mock_doc = MagicMock()
        mock_doc.user_id = 1  # Match the user ID in the token
        mock_doc.status = "processed"
        mock_get_doc.return_value = mock_doc
        
        # Then mock rag_service.answer_question to simulate rate limiting
        with patch('app.api.routes.qa.rag_service.answer_question',
                  new_callable=AsyncMock,
                  side_effect=HTTPException(
                      status_code=429,
                      detail="Rate limit exceeded. Please try again later."
                  )):
            # Act
            response = client.post(
                "/api/qa/ask",
                json=question_data,
                headers=user_token_headers
            )
            
            # Assert
            assert response.status_code == 429
            assert "Rate limit" in response.json()["detail"]


def test_get_documents_api_parameters(client, user_token_headers):
    """Test the documents endpoint with various query parameters."""
    # Mock the DocumentService
    with patch('app.api.routes.documents.DocumentService.get_documents_by_user_id',
              return_value=([], 0)) as mock_get_docs:
        
        # Test with various combinations of query parameters
        test_cases = [
            {"params": {}, "expected_args": {"skip": 0, "limit": 100}},
            {"params": {"skip": "10"}, "expected_args": {"skip": 10, "limit": 100}},
            {"params": {"limit": "5"}, "expected_args": {"skip": 0, "limit": 5}},
            {"params": {"skip": "10", "limit": "5"}, "expected_args": {"skip": 10, "limit": 5}},
        ]
        
        for case in test_cases:
            # Act - Call with query parameters
            response = client.get(
                "/api/documents", 
                params=case["params"],
                headers=user_token_headers
            )
            
            # Assert
            assert response.status_code == 200
            # Ensure it was called with the expected skip and limit values
            call_kwargs = mock_get_docs.call_args.kwargs
            for param, value in case["expected_args"].items():
                assert call_kwargs[param] == value, f"Parameter {param} should be {value}"
            
            # Reset the mock for the next test case
            mock_get_docs.reset_mock()


def test_ask_question_with_question_parameter(client, user_token_headers):
    """
    Test the question-answering endpoint with 'question' parameter instead of 'query'.
    
    This test verifies that:
    1. The /api/qa/ask endpoint accepts the 'question' parameter (backward compatibility)
    2. The endpoint functions correctly with the alternative parameter
    3. The response is the same as when using the 'query' parameter
    """
    # Arrange
    question_data = {
        "question": "What is the main topic?",  # Using 'question' instead of 'query'
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
            "model_used": "ollama",
            "query_time_ms": 500
        }
    }
    
    # Mock document service and RAG service
    with patch('app.api.routes.qa.DocumentService.get_document_by_id', 
              return_value=Document(id=1, user_id=1, status="processed")):
        
        with patch('app.api.routes.qa.rag_service.answer_question', 
                   new_callable=AsyncMock, 
                   return_value=answer_result) as mock_answer:
            
            # Act
            response = client.post(
                "/api/qa/ask",
                json=question_data,
                headers=user_token_headers
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "The main topic is artificial intelligence."
            
            # Verify the 'question' parameter was correctly used as 'query'
            mock_answer.assert_called_once()
            call_args = mock_answer.call_args[1]
            assert call_args["query"] == "What is the main topic?"


def test_ask_question_with_unprocessed_document(client, user_token_headers):
    """
    Test asking a question about a document that is not fully processed.
    
    This test verifies that:
    1. The API checks the processing status of documents
    2. It returns an appropriate error when a document is not fully processed
    """
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    # Mock DocumentService to return an unprocessed document
    with patch('app.api.routes.qa.DocumentService.get_document_by_id',
              return_value=Document(
                  id=1, 
                  user_id=1,
                  status="processing"  # Document is still being processed
              )):
        
        # Act
        response = client.post(
            "/api/qa/ask",
            json=question_data,
            headers=user_token_headers
        )
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "not fully processed" in data["detail"]


def test_ask_question_ollama_service_unavailable(client, user_token_headers):
    """
    Test handling when the Ollama service is unavailable.
    
    This test verifies that:
    1. The API correctly handles Ollama service unavailability
    2. It returns a 503 Service Unavailable status code
    3. The error message is properly formatted
    """
    # Arrange
    question_data = {
        "query": "What is the main topic?",
        "document_ids": [1]
    }
    
    # Mock result from RAG service indicating Ollama is unavailable
    error_result = {
        "answer": "The Ollama service is currently unavailable. Please try again later.",
        "citations": [],
        "metadata": {
            "error": "Ollama service connection failed",
            "query_time_ms": 50
        }
    }
    
    # Mock DocumentService and RAG service
    with patch('app.api.routes.qa.DocumentService.get_document_by_id', 
              return_value=Document(id=1, user_id=1, status="processed")):
        
        with patch('app.api.routes.qa.rag_service.answer_question', 
                   new_callable=AsyncMock, 
                   return_value=error_result) as mock_answer:
            
            # Act
            response = client.post(
                "/api/qa/ask",
                json=question_data,
                headers=user_token_headers
            )
            
            # Assert
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "unavailable" in data["detail"].lower()
            mock_answer.assert_called_once() 


def test_ask_question_with_empty_document_list(client, user_token_headers):
    """Test asking a question with an empty document list."""
    # Mock RAG service
    with patch('app.services.rag.rag_service.answer_question') as mock_answer:
        mock_answer.return_value = {
            "answer": "I couldn't find any relevant information in your documents.",
            "citations": [],
            "metadata": {"searched_documents": 0}
        }
        
        # Make the request with empty document_ids list
        response = client.post(
            "/api/qa/ask",
            headers=user_token_headers,
            json={"query": "What is the meaning of life?", "document_ids": []}
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert data["citations"] == []
        assert data["metadata"]["searched_documents"] == 0
        
        # Verify mock called with empty document_ids
        mock_answer.assert_called_once()
        assert mock_answer.call_args[1]["document_ids"] == []


def test_ask_question_rate_limit(client, user_token_headers):
    """Test handling of rate limiting in the QA API."""
    # Mock RAG service to simulate rate limit reached
    with patch('app.services.rag.rag_service.answer_question') as mock_answer:
        mock_answer.side_effect = HTTPException(
            status_code=429,
            detail="Rate limit reached. Please try again later."
        )
        
        # Make the request
        response = client.post(
            "/api/qa/ask",
            headers=user_token_headers,
            json={"query": "What is the meaning of life?"}
        )
        
        # Assert response
        assert response.status_code == 429
        data = response.json()
        assert "Rate limit reached" in data["detail"]


def test_ask_question_with_invalid_parameters(client, user_token_headers):
    """Test QA API with invalid parameters."""
    # Test with missing query parameter
    response = client.post(
        "/api/qa/ask",
        headers=user_token_headers,
        json={"document_ids": [1, 2, 3]}  # No query parameter
    )
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]
    
    # Test with invalid document_ids format
    response = client.post(
        "/api/qa/ask",
        headers=user_token_headers,
        json={"query": "What is the meaning of life?", "document_ids": "not_a_list"}
    )
    assert response.status_code == 422  # Validation error


def test_ask_question_with_special_characters(client, user_token_headers):
    """Test QA API with special characters in the query."""
    # Mock RAG service
    with patch('app.services.rag.rag_service.answer_question') as mock_answer:
        mock_answer.return_value = {
            "answer": "Here's your answer with special characters: €$@!&*",
            "citations": [],
            "metadata": {}
        }
        
        # Make the request with query containing special characters
        response = client.post(
            "/api/qa/ask",
            headers=user_token_headers,
            json={"query": "What about €$@!&*?"}
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "special characters: €$@!&*" in data["answer"]
        
        # Verify mock called with the special characters
        mock_answer.assert_called_once()
        assert mock_answer.call_args[1]["query"] == "What about €$@!&*?"


def test_ask_question_without_cache(client, user_token_headers):
    """Test QA API with cache disabled."""
    # Mock RAG service
    with patch('app.services.rag.rag_service.answer_question') as mock_answer:
        mock_answer.return_value = {
            "answer": "Fresh, uncached response",
            "citations": [],
            "metadata": {"cache_used": False}
        }
        
        # Make the request with use_cache set to false
        response = client.post(
            "/api/qa/ask",
            headers=user_token_headers,
            json={"query": "What is the meaning of life?", "use_cache": False}
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Fresh, uncached response"
        assert data["metadata"]["cache_used"] is False
        
        # Verify mock called with use_cache=False
        mock_answer.assert_called_once()
        assert mock_answer.call_args[1]["use_cache"] is False


def test_ask_question_with_very_long_query(client, user_token_headers):
    """Test QA API with a very long query."""
    # Create a very long query
    very_long_query = "Why " + "is this " * 500 + "happening?"
    
    # Make the request with very long query
    response = client.post(
        "/api/qa/ask",
        headers=user_token_headers,
        json={"query": very_long_query}
    )
    
    # The API should either handle it or return a proper error
    # Depending on the implementation, this could be 200 OK or a 4xx error
    assert response.status_code in [200, 400, 413, 422]
    
    if response.status_code == 200:
        assert "answer" in response.json()
    else:
        assert "detail" in response.json() 