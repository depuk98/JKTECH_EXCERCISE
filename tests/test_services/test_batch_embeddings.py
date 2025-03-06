import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.document import DocumentService

@pytest.mark.asyncio
async def test_generate_embeddings_batch():
    """Test batch embedding generation."""
    # Test data
    test_texts = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]
    
    # Mock embeddings
    mock_embeddings = [
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist()
    ]
    
    # Use a consistent pattern with patch for mocking
    with patch('app.services.document.embedding_model.encode') as mock_encode, \
         patch('asyncio.get_event_loop') as mock_get_loop:
        # Setup mocks
        mock_encode.return_value = np.array(mock_embeddings)
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.return_value = mock_embeddings
        
        # Call the method
        result = await DocumentService._generate_embeddings_batch(test_texts)
        
        # Verify results
        assert len(result) == len(test_texts)
        for i, embedding in enumerate(result):
            assert len(embedding) == 384  # Expected dimension
            assert isinstance(embedding, list)
            
        # Verify encoder was used
        mock_loop.run_in_executor.assert_called_once()

@pytest.mark.asyncio
async def test_generate_embeddings_batch_with_invalid_inputs():
    """Test batch embedding generation with invalid inputs."""
    # Test data with invalid inputs
    test_texts = [
        "Valid text",
        None,  # Invalid
        "",    # Invalid
        123,   # Invalid
        "Another valid text"
    ]
    
    # Mock embeddings for valid texts
    mock_embeddings = [
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist()
    ]
    
    # Use a consistent pattern with patch for mocking
    with patch('app.services.document.embedding_model.encode') as mock_encode, \
         patch('asyncio.get_event_loop') as mock_get_loop:
        # Setup mocks
        mock_encode.return_value = np.array(mock_embeddings)
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.return_value = mock_embeddings
        
        # Call the method
        result = await DocumentService._generate_embeddings_batch(test_texts)
        
        # Verify results
        assert len(result) == len(test_texts)
        
        # Check valid texts have embeddings
        assert len(result[0]) == 384
        assert len(result[4]) == 384
        
        # Check invalid texts have zero vectors
        assert all(x == 0.0 for x in result[1])
        assert all(x == 0.0 for x in result[2])
        assert all(x == 0.0 for x in result[3])
        
        # Verify encoder was used
        mock_loop.run_in_executor.assert_called_once()

@pytest.mark.asyncio
async def test_generate_embeddings_batch_empty_list():
    """Test batch embedding generation with empty list."""
    # Call the method with empty list
    result = await DocumentService._generate_embeddings_batch([])
    
    # Verify results
    assert result == [] 