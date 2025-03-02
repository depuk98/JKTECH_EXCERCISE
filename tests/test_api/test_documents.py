import io
import json
import pytest
from unittest.mock import patch, MagicMock, ANY

from app.services.document import DocumentService

@pytest.mark.asyncio
async def test_upload_document(client, user_token_headers):
    """Test document upload endpoint."""
    # Create a dummy PDF file
    file_content = b"%PDF-1.7\nTest document content"
    file = io.BytesIO(file_content)
    
    # Mock the DocumentService.upload_document method
    with patch.object(DocumentService, 'upload_document') as mock_upload:
        # Setup the mock to return a document with expected attributes
        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.filename = "test.pdf"
        mock_doc.content_type = "application/pdf"
        mock_doc.status = "pending"
        mock_upload.return_value = mock_doc
        
        # Make the request
        response = client.post(
            "/api/documents/upload",
            headers=user_token_headers,
            files={"file": ("test.pdf", file, "application/pdf")}
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["filename"] == "test.pdf"
        assert data["status"] == "pending"
        
        # Verify mock was called
        mock_upload.assert_called_once()


@pytest.mark.asyncio
async def test_list_documents(client, user_token_headers):
    """Test listing documents endpoint."""
    # Mock the DocumentService.get_documents_by_user_id method
    with patch.object(DocumentService, 'get_documents_by_user_id') as mock_get_docs:
        # Setup the mock to return a list of documents
        mock_docs = [MagicMock() for _ in range(3)]
        for i, doc in enumerate(mock_docs):
            doc.id = i + 1
            doc.filename = f"doc{i+1}.pdf"
            doc.status = "processed"
            doc.content_type = "application/pdf"
            doc.chunks = []
        
        mock_get_docs.return_value = (mock_docs, 3)
        
        # Make the request
        response = client.get(
            "/api/documents?skip=0&limit=10",
            headers=user_token_headers
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["documents"]) == 3
        
        # Verify mock was called with correct params
        mock_get_docs.assert_called_once()
        args, kwargs = mock_get_docs.call_args
        assert kwargs["skip"] == 0
        assert kwargs["limit"] == 10


@pytest.mark.asyncio
async def test_search_documents(client, user_token_headers):
    """Test document search endpoint."""
    # Mock the DocumentService.search_documents method
    with patch.object(DocumentService, 'search_documents') as mock_search:
        # Setup the mock to return search results
        mock_chunk1 = MagicMock()
        mock_chunk1.id = 1
        mock_chunk1.document_id = 1
        mock_chunk1.text = "This is a test document"
        mock_chunk1.chunk_metadata = {"page": 1}
        
        mock_chunk2 = MagicMock()
        mock_chunk2.id = 2
        mock_chunk2.document_id = 1
        mock_chunk2.text = "More test content"
        mock_chunk2.chunk_metadata = {"page": 2}
        
        # Return mock chunks with similarity scores
        mock_search.return_value = [
            (mock_chunk1, 0.85),
            (mock_chunk2, 0.75)
        ]
        
        # Make the request
        response = client.get(
            "/api/documents/search?query=test&limit=5",
            headers=user_token_headers
        )
        
        # Assert response
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        assert results[0]["chunk_id"] == 1
        assert results[0]["similarity_score"] == 0.85
        assert results[1]["chunk_id"] == 2
        assert results[1]["similarity_score"] == 0.75
        
        # Verify mock was called with correct params
        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        assert kwargs["query"] == "test"
        assert kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_get_document(client, user_token_headers):
    """Test getting a document by ID."""
    # Mock the DocumentService.get_document_by_id method
    with patch.object(DocumentService, 'get_document_by_id') as mock_get_doc:
        # Setup the mock to return a document
        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.user_id = 1  # Same as the user in the token
        mock_doc.filename = "test.pdf"
        mock_doc.content_type = "application/pdf"
        mock_doc.status = "processed"
        mock_doc.chunks = []
        
        mock_get_doc.return_value = mock_doc
        
        # Make the request
        response = client.get(
            "/api/documents/1",
            headers=user_token_headers
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["filename"] == "test.pdf"
        
        # Verify mock was called
        mock_get_doc.assert_called_once_with(db=ANY, document_id=1)


@pytest.mark.asyncio
async def test_delete_document(client, user_token_headers):
    """Test deleting a document."""
    # Mock the DocumentService.delete_document method
    with patch.object(DocumentService, 'delete_document') as mock_delete:
        # Setup the mock to return success
        mock_delete.return_value = True
        
        # Make the request
        response = client.delete(
            "/api/documents/1",
            headers=user_token_headers
        )
        
        # Assert response (204 No Content)
        assert response.status_code == 204
        
        # Verify mock was called with correct params
        mock_delete.assert_called_once()
        args, kwargs = mock_delete.call_args
        assert kwargs["document_id"] == 1


@pytest.mark.asyncio
async def test_delete_document_not_found(client, user_token_headers):
    """Test deleting a document that doesn't exist."""
    # Mock the DocumentService.delete_document method
    with patch.object(DocumentService, 'delete_document') as mock_delete:
        # Setup the mock to return failure (document not found)
        mock_delete.return_value = False
        
        # Make the request
        response = client.delete(
            "/api/documents/999",
            headers=user_token_headers
        )
        
        # Assert response (404 Not Found)
        assert response.status_code == 404
        
        # Verify mock was called
        mock_delete.assert_called_once_with(db=ANY, document_id=999, user_id=ANY) 