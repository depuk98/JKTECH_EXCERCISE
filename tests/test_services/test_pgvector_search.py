import pytest
import json
from unittest.mock import patch, MagicMock
import numpy as np
from sqlalchemy import text

from app.services.document import DocumentService
from app.models.document import DocumentChunk


@pytest.fixture
def mock_embedding():
    """Return a mock embedding vector of dimension 384"""
    return np.random.rand(384).tolist()


@pytest.mark.asyncio
async def test_generate_embeddings():
    """Test embedding generation function."""
    test_text = "This is a test document for embedding generation."
    
    # Execute the function
    embeddings = await DocumentService._generate_embeddings(test_text)
    
    # Verify the embeddings
    assert isinstance(embeddings, list)
    assert len(embeddings) == 384  # Expected dimension for the all-MiniLM-L6-v2 model
    
    # Verify all values are floats
    for value in embeddings:
        assert isinstance(value, float)


@pytest.mark.asyncio
async def test_vector_search_with_pgvector(db, mock_embedding):
    """Test vector search using pgvector."""
    # Mock the necessary elements for testing
    user_id = 1
    query = "test query"
    limit = 5
    
    # Mock _generate_embeddings to return a fixed embedding vector
    with patch.object(
        DocumentService, '_generate_embeddings', return_value=mock_embedding
    ) as mock_generate:
        
        # Mock the SQLAlchemy execute to simulate pgvector results
        mock_result = [
            MagicMock(id=1, document_id=1, chunk_index=0, text="Result 1", 
                     chunk_metadata=json.dumps({"page": 1}), created_at="2023-01-01", 
                     similarity_score=0.85),
            MagicMock(id=2, document_id=1, chunk_index=1, text="Result 2", 
                     chunk_metadata=json.dumps({"page": 2}), created_at="2023-01-01", 
                     similarity_score=0.75),
        ]
        
        with patch.object(db, 'execute') as mock_execute:
            # Setup the mock for execute to return our mock results
            mock_execute.return_value = mock_result
            
            # Call the search function
            results = await DocumentService.search_documents(
                db=db, user_id=user_id, query=query, limit=limit
            )
            
            # Verify embedding generation was called
            mock_generate.assert_called_once_with(query)
            
            # Verify execute was called (not checking exact SQL as it's complex)
            mock_execute.assert_called()
            
            # Verify results format
            assert len(results) == len(mock_result)


@pytest.mark.asyncio
async def test_fallback_to_keyword_search(db, mock_embedding):
    """Test fallback to keyword search when vector search fails."""
    user_id = 1
    query = "test query"
    limit = 5
    
    # Mock _generate_embeddings to return a fixed embedding vector
    with patch.object(
        DocumentService, '_generate_embeddings', return_value=mock_embedding
    ) as mock_generate:
        
        # Mock execute to fail first on vector search then succeed for keyword search
        with patch.object(db, 'execute') as mock_execute:
            # First call (vector search) raises an exception
            mock_execute.side_effect = [
                Exception("Vector search failed"),
                # Second call (keyword search) returns results
                [
                    MagicMock(id=1, document_id=1, chunk_index=0, text="Keyword Result 1", 
                             chunk_metadata=json.dumps({"page": 1}), created_at="2023-01-01"),
                    MagicMock(id=2, document_id=1, chunk_index=1, text="Keyword Result 2", 
                             chunk_metadata=json.dumps({"page": 2}), created_at="2023-01-01")
                ]
            ]
            
            # Mock rollback to avoid errors from mock
            db.rollback = MagicMock()
            
            # Call the search function
            results = await DocumentService.search_documents(
                db=db, user_id=user_id, query=query, limit=limit
            )
            
            # Verify embedding generation was called
            mock_generate.assert_called_once_with(query)
            
            # Verify db rollback was called after vector search failure
            db.rollback.assert_called_once()
            
            # Verify execute was called at least twice (once for vector, once for keyword)
            assert mock_execute.call_count >= 2
            
            # Verify results were returned from keyword search
            assert len(results) == 2


@pytest.mark.asyncio
async def test_vector_search_pgvector_syntax():
    """Test the SQL syntax used for pgvector search is correct."""
    # This test doesn't execute the query but verifies the SQL syntax used with pgvector
    with patch('sqlalchemy.text') as mock_text:
        # Mock text to track SQL statements
        mock_text.side_effect = lambda x: x
        
        # Track all SQL statements passed to text()
        sql_statements = []
        def capture_sql(sql):
            sql_statements.append(sql)
            return sql
            
        mock_text.side_effect = capture_sql
        
        with patch.object(DocumentService, '_generate_embeddings', return_value=[0.1, 0.2, 0.3]):
            with patch.object(MagicMock(), 'execute'):
                # Call method to generate SQL 
                try:
                    await DocumentService.search_documents(
                        db=MagicMock(), user_id=1, query="test", limit=5
                    )
                except:
                    pass
                
                # Check that we have SQL statements containing our expected patterns
                pgvector_function_sql = next((s for s in sql_statements if "pgvector_parse_array" in s and "CREATE OR REPLACE" in s), None)
                vector_search_sql = next((s for s in sql_statements if "cosine_distance" in s or "<=> " in s or "pgvector_parse_array" in s and "SELECT" in s), None)
                
                # Verify we have both SQL statements
                assert pgvector_function_sql is not None, "SQL with pgvector_parse_array function creation not found"
                assert vector_search_sql is not None, "Vector search SQL not found"


@pytest.mark.asyncio
async def test_create_vector_function(db):
    """Test that the pgvector_parse_array function is created in PostgreSQL."""
    # Mock the execute function to simulate function creation
    with patch.object(db, 'execute') as mock_execute:
        with patch.object(db, 'commit') as mock_commit:
            # Call a function that would create the custom function
            await DocumentService.search_documents(db=db, user_id=1, query="test", limit=5)
            
            # Check that the execute function was called
            mock_execute.assert_called()
            
            # Check that commit was called after function creation
            mock_commit.assert_called() 