import pytest
from unittest.mock import MagicMock, patch, call
from sqlalchemy.orm import Session

from app.services.document import DocumentService
from app.models.document import Document, DocumentChunk
from app.models.user import User


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_document():
    """Create a mock document."""
    document = MagicMock(spec=Document)
    document.id = 1
    document.user_id = 1
    document.filename = "test_document.pdf"
    document.content_type = "application/pdf"
    document.status = "processed"
    return document


@pytest.fixture
def mock_document_chunks():
    """Create mock document chunks."""
    chunks = []
    for i in range(3):
        chunk = MagicMock(spec=DocumentChunk)
        chunk.id = i + 1
        chunk.document_id = 1
        chunk.chunk_index = i
        chunk.text = f"Test chunk {i+1}"
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_delete_document_success(mock_db, mock_document):
    """Test successful document deletion."""
    document_id = 1
    user_id = 1
    
    # Reset the mock to clear previous calls
    mock_db.query.reset_mock()
    
    # Setup mock query result for document
    mock_db.query.return_value.filter.return_value.first.return_value = mock_document
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that the document was queried (may not be the first call)
    assert any(call(Document) in mock_db.query.call_args_list for call in [call])
    
    # Assert that a delete operation occurred
    mock_db.delete.assert_called_once_with(mock_document)
    
    # Assert that the transaction was committed
    mock_db.commit.assert_called_once()
    
    # Assert function returned success
    assert result is True


@pytest.mark.asyncio
async def test_delete_document_not_found(mock_db):
    """Test deletion of non-existent document."""
    document_id = 999
    user_id = 1
    
    # Setup mock query result for non-existent document
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that query was executed but no document was found
    mock_db.query.assert_called_with(Document)
    mock_db.query.return_value.filter.assert_called_once()
    
    # Assert that no delete operation occurred
    mock_db.delete.assert_not_called()
    
    # Assert that the transaction was not committed
    mock_db.commit.assert_not_called()
    
    # Assert function returned failure
    assert result is False


@pytest.mark.asyncio
async def test_delete_document_wrong_user(mock_db, mock_document):
    """Test deletion when document belongs to a different user."""
    document_id = 1
    user_id = 2  # Different from document.user_id which is 1
    
    # Create a document belonging to a different user
    mock_document.user_id = 1
    
    # Setup mock query result
    mock_db.query.return_value.filter.return_value.first.return_value = mock_document
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that the document was found
    mock_db.query.assert_called_with(Document)
    
    # Assert that no delete operation occurred
    mock_db.delete.assert_not_called()
    
    # Assert that the transaction was not committed
    mock_db.commit.assert_not_called()
    
    # Assert function returned failure
    assert result is False


@pytest.mark.asyncio
async def test_delete_document_with_exception(mock_db, mock_document):
    """Test exception handling during document deletion."""
    document_id = 1
    user_id = 1
    
    # Setup mock query result
    mock_db.query.return_value.filter.return_value.first.return_value = mock_document
    
    # Setup mock to raise exception during delete
    mock_db.delete.side_effect = Exception("Database error")
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that rollback was called
    mock_db.rollback.assert_called_once()
    
    # Assert function returned failure
    assert result is False


@pytest.mark.asyncio
async def test_delete_document_cascades_to_chunks(mock_db, mock_document, mock_document_chunks):
    """Test that deleting a document also cascades to its chunks."""
    document_id = 1
    user_id = 1
    
    # Setup mocks for cascade testing
    mock_db.query.return_value.filter.return_value.first.return_value = mock_document
    
    # Configure another query mock for chunks to verify they're also deleted
    mock_db.query.return_value.filter.return_value.all.return_value = mock_document_chunks
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that chunks were queried 
    chunk_query_called = False
    for call in mock_db.query.call_args_list:
        if call[0][0] == DocumentChunk:
            chunk_query_called = True
            break
    
    # In the actual code, cascade delete should handle this automatically,
    # but this test verifies the logic behind it
    assert chunk_query_called or mock_db.query.return_value.filter.return_value.delete.called
    
    # Assert that the document was deleted
    mock_db.delete.assert_called_once_with(mock_document)
    
    # Assert function returned success
    assert result is True 