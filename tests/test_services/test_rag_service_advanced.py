import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import httpx

from app.services.rag import RAGService, LANGCHAIN_AVAILABLE
from app.models.document import Document, DocumentChunk
from app.services.document import DocumentService


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.refresh = MagicMock()
    db.query = MagicMock()
    db.execute = MagicMock()
    return db


@pytest.fixture
def rag_service():
    """Create a RAG service instance."""
    return RAGService()


@pytest.mark.asyncio
async def test_check_ollama_available_success(rag_service):
    """Test successful Ollama availability check."""
    # Mock both LangChain and direct API
    if LANGCHAIN_AVAILABLE:
        with patch('langchain_ollama.OllamaLLM.invoke', return_value="Hello, I'm an AI assistant.") as mock_invoke:
            # Check availability
            result = await rag_service.check_ollama_available()
            
            # Verify result
            assert result is True
            mock_invoke.assert_called_once()
    else:
        # Mock successful response from Ollama
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": "Hello, I'm an AI assistant."}
            mock_post.return_value = mock_response
            
            # Check availability
            result = await rag_service.check_ollama_available()
            
            # Verify result
            assert result is True
            mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_check_ollama_available_connection_error(rag_service):
    """Test Ollama availability check with connection error."""
    # Mock both LangChain and direct API in the same test
    with patch('langchain_ollama.OllamaLLM.invoke', side_effect=Exception("Connection refused")), \
         patch('httpx.AsyncClient.post', side_effect=httpx.ConnectError("Connection refused")):
        
        # Check availability
        result = await rag_service.check_ollama_available()
        
        # Verify result
        assert result is False


@pytest.mark.asyncio
async def test_check_ollama_available_invalid_response(rag_service):
    """Test Ollama availability check with invalid response."""
    # Mock both LangChain and direct API in the same test
    with patch('langchain_ollama.OllamaLLM.invoke', return_value=None), \
         patch('httpx.AsyncClient.post') as mock_post:
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "Model not found"}  # Missing "response" key
        mock_post.return_value = mock_response
        
        # Check availability
        result = await rag_service.check_ollama_available()
        
        # Verify result
        assert result is False


def test_count_tokens(rag_service):
    """Test token counting functionality."""
    # Test with various text inputs
    empty_text = ""
    short_text = "Hello, world!"
    long_text = "This is a longer text with multiple sentences. It contains punctuation, numbers (123), and special characters like @#$%."
    
    # Count tokens
    empty_count = rag_service.count_tokens(empty_text)
    short_count = rag_service.count_tokens(short_text)
    long_count = rag_service.count_tokens(long_text)
    
    # Verify counts are reasonable
    assert empty_count == 0
    assert short_count > 0
    assert long_count > short_count
    
    # Verify punctuation is counted
    text_with_punctuation = "Hello, world!"
    text_without_punctuation = "Hello world"
    assert rag_service.count_tokens(text_with_punctuation) > rag_service.count_tokens(text_without_punctuation)


@pytest.mark.asyncio
async def test_retrieve_context_with_document_ids_sql_error(mock_db, rag_service):
    """Test error handling in retrieve_context when SQL execution fails."""
    # Mock SQL error
    mock_db.execute.side_effect = Exception("SQL error")
    
    # Call retrieve_context
    result = await rag_service.retrieve_context(
        db=mock_db,
        user_id=1,
        query="test query",
        document_ids=[1, 2],
        top_k=5
    )
    
    # Verify empty result is returned
    assert result == []


def test_format_context_with_empty_input(rag_service):
    """Test format_context with empty input."""
    # Call with empty input
    formatted_context, citations = rag_service.format_context([])
    
    # Verify empty output
    assert formatted_context == ""
    assert citations == []


def test_format_context_truncation(rag_service):
    """Test that format_context truncates very long chunks."""
    # Create a context chunk with very long text
    long_chunk = {
        "chunk_id": 1,
        "document_id": 1,
        "filename": "long_doc.pdf",
        "text": "A" * 2000,  # 2000 characters
        "metadata": {"page": 1},
        "similarity_score": 0.9
    }
    
    # Format the context
    formatted_context, citations = rag_service.format_context([long_chunk])
    
    # Verify truncation
    assert len(formatted_context) < 2000 + 10  # Adding some buffer for citation marker
    assert "..." in formatted_context  # Should contain truncation indicator
    assert citations[0]["id"] == 1
    assert citations[0]["filename"] == "long_doc.pdf"


def test_format_context_token_limit(rag_service):
    """Test that format_context respects token limit for very large contexts."""
    # Create multiple large chunks that exceed the token limit
    large_chunks = []
    for i in range(10):
        large_chunks.append({
            "chunk_id": i,
            "document_id": i,
            "filename": f"doc{i}.pdf",
            "text": f"This is document {i}. " + "A" * 5000,  # Very large text
            "metadata": {"page": 1},
            "similarity_score": 0.9 - (i * 0.05)  # Decreasing relevance
        })
    
    # Patch the count_tokens method to return a large value
    with patch.object(rag_service, 'count_tokens', side_effect=lambda x: len(x)):
        with patch.object(rag_service, 'max_tokens', 10000):  # Set a smaller limit for testing
            # Format the context
            formatted_context, citations = rag_service.format_context(large_chunks)
            
            # Verify the context was truncated
            assert len(formatted_context) <= 10000
            assert len(citations) < 10  # Some chunks should be removed


def test_generate_prompt(rag_service):
    """Test prompt generation with various inputs."""
    # Test with different queries and contexts
    short_query = "What is AI?"
    long_query = "Can you explain the differences between supervised, unsupervised, and reinforcement learning in detail?"
    short_context = "[1] AI stands for Artificial Intelligence."
    long_context = "[1] Artificial Intelligence (AI) refers to systems that can perform tasks requiring human intelligence.\n\n" + \
                  "[2] Machine Learning is a subset of AI focused on learning from data.\n\n" + \
                  "[3] Deep Learning is a subset of Machine Learning using neural networks with many layers."
    
    # Generate prompts
    prompt1 = rag_service.generate_prompt(short_query, short_context)
    prompt2 = rag_service.generate_prompt(long_query, long_context)
    
    # Verify prompts contain all necessary components
    for prompt in [prompt1, prompt2]:
        assert "CONTEXT INFORMATION:" in prompt
        assert "USER QUESTION:" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "citation markers" in prompt.lower()
        assert "Your answer:" in prompt
    
    # Verify query and context are included
    assert short_query in prompt1
    assert short_context in prompt1
    assert long_query in prompt2
    assert long_context in prompt2


@pytest.mark.asyncio
async def test_generate_answer_ollama_connection_error(rag_service):
    """Test error handling when Ollama connection fails."""
    # Mock both LangChain and direct API
    if LANGCHAIN_AVAILABLE:
        with patch('langchain_ollama.OllamaLLM.invoke', side_effect=Exception("Connection refused")), \
             patch('httpx.AsyncClient.post', side_effect=httpx.ConnectError("Connection refused")):
            
            # Generate answer
            result = await rag_service.generate_answer_ollama("Test prompt")
            
            # Verify error message
            assert "Could not connect to Ollama" in result
    else:
        # Mock connection error
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            
            # Generate answer
            result = await rag_service.generate_answer_ollama("Test prompt")
            
            # Verify error message
            assert "Could not connect to Ollama" in result


@pytest.mark.asyncio
async def test_generate_answer_ollama_http_error(rag_service):
    """Test error handling when Ollama returns HTTP error."""
    # Mock both LangChain and direct API
    if LANGCHAIN_AVAILABLE:
        with patch('langchain_ollama.OllamaLLM.invoke', side_effect=Exception("404 Not Found")), \
             patch('httpx.AsyncClient.post') as mock_post:
            
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", 
                request=MagicMock(), 
                response=MagicMock(status_code=404, text="Model not found")
            )
            mock_post.return_value = mock_response
            
            # Generate answer
            result = await rag_service.generate_answer_ollama("Test prompt")
            
            # Verify error message contains either "Model" or "not found"
            assert any(text in result for text in ["Model", "not found"])
    else:
        # Mock HTTP error
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", 
                request=MagicMock(), 
                response=MagicMock(status_code=404, text="Model not found")
            )
            mock_post.return_value = mock_response
            
            # Generate answer
            result = await rag_service.generate_answer_ollama("Test prompt")
            
            # Verify error message contains either "Model" or "not found"
            assert any(text in result for text in ["Model", "not found"])


@pytest.mark.asyncio
async def test_generate_answer_ollama_empty_response(rag_service):
    """Test error handling when Ollama returns empty response."""
    # Mock both LangChain and direct API
    if LANGCHAIN_AVAILABLE:
        with patch('langchain_ollama.OllamaLLM.invoke', return_value=""), \
             patch('httpx.AsyncClient.post') as mock_post:
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": ""}
            mock_post.return_value = mock_response
            
            # Generate answer
            result = await rag_service.generate_answer_ollama("Test prompt")
            
            # Verify error message
            assert any(text in result for text in ["Sorry", "error", "empty"])
    else:
        # Mock empty response
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"response": ""}
            mock_post.return_value = mock_response
            
            # Generate answer
            result = await rag_service.generate_answer_ollama("Test prompt")
            
            # Verify error message
            assert any(text in result for text in ["Sorry", "error", "empty"])


def test_cache_operations(rag_service):
    """Test caching operations."""
    # Generate a cache key
    cache_key = rag_service.generate_cache_key("test query", 1, [1, 2])
    
    # Create test response
    test_response = {
        "answer": "This is a test answer",
        "citations": [{"id": 1, "filename": "test.pdf"}],
        "metadata": {"processing_time": 0.5}
    }
    
    # Store in cache
    rag_service.store_cache_response(cache_key, test_response)
    
    # Retrieve from cache
    cached_response = rag_service.get_cached_response(cache_key)
    
    # Verify cache hit
    assert cached_response is not None
    assert cached_response["answer"] == "This is a test answer"
    
    # Test cache key generation with different inputs
    key1 = rag_service.generate_cache_key("query", 1, [1, 2])
    key2 = rag_service.generate_cache_key("query", 1, [2, 1])  # Same docs, different order
    key3 = rag_service.generate_cache_key("query", 2, [1, 2])  # Different user
    key4 = rag_service.generate_cache_key("different query", 1, [1, 2])  # Different query
    
    # Verify keys are consistent for same inputs but different for different inputs
    assert key1 == key2  # Order of document IDs shouldn't matter
    assert key1 != key3  # Different user should have different key
    assert key1 != key4  # Different query should have different key


@pytest.mark.asyncio
async def test_answer_question_with_cache(mock_db, rag_service):
    """Test answer_question with cache hit."""
    # Create a cache entry
    query = "test query"
    cache_key = rag_service.generate_cache_key(query, 1)
    cached_response = {
        "answer": "Cached answer",
        "citations": [],
        "metadata": {"processing_time": 0.1}
    }
    rag_service.store_cache_response(cache_key, cached_response)
    
    # Call answer_question
    result = await rag_service.answer_question(
        db=mock_db,
        user_id=1,
        query=query,
        use_cache=True
    )
    
    # Verify cached response is returned
    assert result["answer"] == "Cached answer"
    assert result["metadata"]["cached"] is True


@pytest.mark.asyncio
async def test_answer_question_ollama_unavailable(mock_db, rag_service):
    """Test answer_question when Ollama is unavailable."""
    # Mock Ollama unavailable
    with patch.object(rag_service, 'check_ollama_available', return_value=False):
        # Call answer_question
        result = await rag_service.answer_question(
            db=mock_db,
            user_id=1,
            query="test query"
        )
        
        # Verify error response
        assert "AI service is not available" in result["answer"]
        assert "error" in result["metadata"]


@pytest.mark.asyncio
async def test_answer_question_with_error_in_generate_answer(mock_db, rag_service):
    """Test answer_question when generate_answer_ollama raises an exception."""
    # Clear the cache to ensure we don't get a cached response
    from app.services.rag import QUERY_CACHE
    QUERY_CACHE.clear()
    
    # Mock Ollama available but generate_answer fails
    with patch.object(rag_service, 'check_ollama_available', return_value=True), \
         patch.object(rag_service, 'retrieve_context', return_value=[{"chunk_id": 1, "text": "test"}]), \
         patch.object(rag_service, 'format_context', return_value=("formatted context", [])), \
         patch.object(rag_service, 'generate_answer_ollama', side_effect=Exception("LLM error")):
        
        # Call answer_question with a unique query to avoid cache hits
        result = await rag_service.answer_question(
            db=mock_db,
            user_id=1,
            query=f"test query {datetime.now().timestamp()}",
            use_cache=False  # Explicitly disable cache
        )
        
        # Verify error response
        assert any(text in result["answer"] for text in ["Sorry", "AI service", "unavailable"])
        assert "error" in result["metadata"] or "LLM error" in result["metadata"].get("error", "")


@pytest.mark.asyncio
async def test_get_available_documents(mock_db, rag_service):
    """Test get_available_documents method."""
    # Mock document service
    with patch.object(
        DocumentService, 'get_documents_by_user_id',
        return_value=([
            Document(
                id=1,
                user_id=1,
                filename="doc1.pdf",
                content_type="application/pdf",
                status="processed"
            ),
            Document(
                id=2,
                user_id=1,
                filename="doc2.pdf",
                content_type="application/pdf",
                status="error"  # This one should be excluded
            )
        ], 2)
    ):
        # Get available documents
        result = await rag_service.get_available_documents(
            db=mock_db,
            user_id=1
        )
        
        # Verify only processed documents are returned
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["filename"] == "doc1.pdf" 