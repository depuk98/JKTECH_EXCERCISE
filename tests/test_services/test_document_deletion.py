import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from app.services.document import DocumentService
from app.models.document import Document, DocumentChunk
from app.models.user import User


@pytest.fixture
def mock_async_db():
    """Create a mock async database session."""
    mock = AsyncMock(spec=AsyncSession)
    # Set up execute to return a result with scalars method
    execute_result = AsyncMock()
    execute_result.scalars = MagicMock()
    execute_result.scalars.return_value.first = MagicMock()
    
    mock.execute.return_value = execute_result
    return mock


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
async def test_delete_document_success(mock_async_db, mock_document):
    """Test successful document deletion."""
    document_id = 1
    user_id = 1
    
    # Setup mock query result for document
    execute_result = mock_async_db.execute.return_value
    execute_result.scalars.return_value.first.return_value = mock_document
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_async_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that execute was called with a select statement
    mock_async_db.execute.assert_called()
    
    # Assert that delete was called with the document
    mock_async_db.delete.assert_called_once_with(mock_document)
    
    # Assert that commit was called
    mock_async_db.commit.assert_called_once()
    
    # Assert function returned success
    assert result is True


@pytest.mark.asyncio
async def test_delete_document_not_found(mock_async_db):
    """Test deletion of non-existent document."""
    document_id = 999
    user_id = 1
    
    # Setup mock query result for non-existent document
    execute_result = mock_async_db.execute.return_value
    execute_result.scalars.return_value.first.return_value = None
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_async_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that execute was called
    mock_async_db.execute.assert_called_once()
    
    # Assert that delete was not called
    mock_async_db.delete.assert_not_called()
    
    # Assert that commit was not called
    mock_async_db.commit.assert_not_called()
    
    # Assert function returned failure
    assert result is False


@pytest.mark.asyncio
async def test_delete_document_wrong_user(mock_async_db, mock_document):
    """Test deletion when document belongs to a different user."""
    document_id = 1
    user_id = 2  # Different from document.user_id which is 1
    
    # Create a document belonging to a different user
    mock_document.user_id = 1
    
    # Setup mock query result
    execute_result = mock_async_db.execute.return_value
    execute_result.scalars.return_value.first.return_value = mock_document
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_async_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that execute was called
    mock_async_db.execute.assert_called_once()
    
    # Assert that delete was not called
    mock_async_db.delete.assert_not_called()
    
    # Assert that commit was not called
    mock_async_db.commit.assert_not_called()
    
    # Assert function returned failure
    assert result is False


@pytest.mark.asyncio
async def test_delete_document_with_exception(mock_async_db, mock_document):
    """Test exception handling during document deletion."""
    document_id = 1
    user_id = 1
    
    # Setup mock query result
    execute_result = mock_async_db.execute.return_value
    execute_result.scalars.return_value.first.return_value = mock_document
    
    # Setup mock to raise exception during delete
    mock_async_db.delete.side_effect = Exception("Database error")
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_async_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that rollback was called
    mock_async_db.rollback.assert_called_once()
    
    # Assert function returned failure
    assert result is False


@pytest.mark.asyncio
async def test_delete_document_cascades_to_chunks(mock_async_db, mock_document):
    """Test that deleting a document also cascades to its chunks."""
    document_id = 1
    user_id = 1
    
    # Setup mocks for cascade testing
    execute_result = mock_async_db.execute.return_value
    execute_result.scalars.return_value.first.return_value = mock_document
    
    # Call the delete function
    result = await DocumentService.delete_document(
        db=mock_async_db,
        document_id=document_id,
        user_id=user_id
    )
    
    # Assert that the document was deleted
    mock_async_db.delete.assert_called_once_with(mock_document)
    
    # Assert that the SQL execute method was called to delete chunks
    # The method should be called at least twice - once for document query and once for chunk deletion
    assert mock_async_db.execute.call_count >= 2
    
    # Assert function returned success
    assert result is True 