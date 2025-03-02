import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock, ANY
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.document import DocumentService
from app.models.document import Document, DocumentChunk
from app.models.user import User


@pytest.fixture
def mock_db():
    return MagicMock(spec=AsyncSession)


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = 1
    user.email = "test@example.com"
    user.username = "testuser"
    return user


@pytest.fixture
def test_pdf_file():
    content = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test PDF Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000120 00000 n\n0000000210 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n310\n%%EOF"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(content)
    temp_file.close()
    
    yield temp_file.name
    
    os.unlink(temp_file.name)


@pytest.fixture
def mock_upload_file(test_pdf_file):
    with open(test_pdf_file, "rb") as f:
        content = f.read()

    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "test_document.pdf"
    upload_file.content_type = "application/pdf"
    
    # Create the file attribute properly
    upload_file.file = MagicMock()
    upload_file.file.read = MagicMock(return_value=content)
    upload_file.seek = MagicMock()
    
    return upload_file


@pytest.mark.asyncio
async def test_upload_document(mock_db, mock_user, mock_upload_file):
    # Mock db add and commit
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()
    mock_db.refresh = MagicMock()
    
    # Mock create_task
    with patch('asyncio.create_task') as mock_create_task:
        document = await DocumentService.upload_document(
            db=mock_db,
            user=mock_user,
            file=mock_upload_file
        )
        
        # Verify document was created with correct attributes
        assert document.user_id == mock_user.id
        assert document.filename == "test_document.pdf"
        assert document.content_type == "application/pdf"
        assert document.status == "pending"
        
        # Verify db operations were called
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        
        # Verify background processing was started
        mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_get_document_by_id(mock_db):
    # Mock db query execution
    mock_document = MagicMock(spec=Document)
    mock_document.id = 1
    mock_document.user_id = 1
    
    # Set up the mock database to handle the query correctly
    mock_db.query = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_document
    
    # Test retrieval
    result = await DocumentService.get_document_by_id(mock_db, document_id=1)
    
    # Verify result
    assert result is mock_document
    mock_db.query.assert_called_with(Document)
    mock_db.query.return_value.filter.assert_called_once()


@pytest.mark.asyncio
async def test_get_documents_by_user_id(mock_db):
    # Mock result set
    mock_documents = [MagicMock(spec=Document) for _ in range(3)]
    mock_total = 3
    
    # Set up the mock database correctly
    mock_db.query = MagicMock()
    mock_db.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = mock_documents
    mock_db.query.return_value.filter.return_value.scalar.return_value = mock_total
    
    # Test retrieval
    result, total = await DocumentService.get_documents_by_user_id(
        mock_db, user_id=1, skip=0, limit=10
    )
    
    # Verify results
    assert result == mock_documents
    assert total == mock_total
    
    # Verify query was called with Document
    mock_db.query.assert_called_with(Document)


@pytest.mark.asyncio
async def test_search_documents(mock_db):
    # Mock search results
    mock_search_results = [
        (MagicMock(spec=DocumentChunk), 0.85),
        (MagicMock(spec=DocumentChunk), 0.75),
    ]
    
    # Mock embedding generation
    with patch.object(
        DocumentService, '_generate_embeddings',
        return_value=[0.1, 0.2, 0.3]
    ) as mock_generate:
        # Mock db execute
        mock_db.execute = MagicMock()
        mock_db.execute.return_value = mock_search_results
        
        # Test search
        results = await DocumentService.search_documents(
            mock_db, user_id=1, query="test query", limit=2
        )
        
        # Verify embedding was generated
        mock_generate.assert_called_once_with("test query")
        
        # Verify execute was called (don't check exact number of calls)
        assert mock_db.execute.called
        
        # Verify results
        assert len(results) == 2 