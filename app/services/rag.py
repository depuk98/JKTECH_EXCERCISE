"""
RAG (Retrieval Augmented Generation) service for document Q&A.

This service implements a Retrieval Augmented Generation (RAG) pipeline:
1. Retrieval: Find relevant document chunks based on semantic similarity to the query
2. Augmentation: Format the retrieved chunks into a context prompt
3. Generation: Use Ollama's LLM to generate an answer based on the context

The process flow is:
- User submits a question
- System retrieves relevant document chunks using vector similarity search
- Retrieved chunks are formatted into a context prompt
- The prompt is sent to Ollama's LLM (llama3.2)
- LLM generates an answer based on the context
- Answer is returned with citations to source documents
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
import re  # For simple token counting

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import text  # For SQL queries
from sqlalchemy.ext.asyncio import AsyncSession

# LangChain imports for Ollama integration
try:
    from langchain_ollama import OllamaLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning(
        "langchain_community not installed. Falling back to direct API calls. "
        "To install: pip install langchain-community"
    )

from app.core.config import settings
from app.models.document import DocumentChunk
from app.services.document import DocumentService

# Configure logger
logger = logging.getLogger(__name__)

# In-memory cache for query responses
QUERY_CACHE = {}
CACHE_EXPIRY = timedelta(minutes=30)  # Cache expires after 30 minutes

# Cache implementation notes:
# - Simple in-memory cache for query responses
# - Cache key is based on user ID, query hash, and document IDs
# - Entries expire after 30 minutes
# - For production use, consider using Redis or another distributed cache
# - Cache is cleared on server restart

class RAGService:
    """
    Service for performing Retrieval Augmented Generation (RAG) for document Q&A.
    Uses Ollama API to generate answers based on retrieved document context.
    """
    
    def __init__(self):
        """Initialize the RAG service."""
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.LLM_MODEL  # Ollama model to use
        self.max_tokens = 4096  # Max context tokens for the model
        self.token_limit = 4000  # Response token limit
        self.temperature = 0.7
        self.answer_max_tokens = 1000  # Max tokens for answer generation
        
        # Log initialization and integration method
        logger.info(f"RAG Service initialized with model: {self.model_name}, URL: {self.ollama_base_url}")
        if LANGCHAIN_AVAILABLE:
            logger.info("Using LangChain integration for Ollama API (preferred)")
        else:
            logger.info("Using direct API calls for Ollama API (fallback)")
    
    async def check_ollama_available(self) -> bool:
        """
        Check if Ollama server is available and the configured model exists.
        
        Returns:
            True if Ollama is available and model exists, False otherwise
        """
        try:
            base_url = self.ollama_base_url.rstrip("/")
            
            # Using langchain if available
            if LANGCHAIN_AVAILABLE:
                try:
                    # Just initialize the client to check connection
                    ollama = OllamaLLM(base_url=base_url, model=self.model_name)
                    # Simple test with a quick prompt
                    result = ollama.invoke("Hello, how are you?")
                    if result and isinstance(result, str):
                        logger.info(f"Ollama is available via LangChain with model '{self.model_name}'")
                        return True
                    else:
                        logger.warning(f"Unexpected response from LangChain Ollama: {result}")
                except Exception as e:
                    logger.error(f"Error using LangChain Ollama: {str(e)}")
                    # Fall through to direct API
            
            # Direct API testing - exactly matching the successful curl pattern
            logger.info("Trying direct API call to Ollama")
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    # Test request exactly matching the curl command pattern
                    test_payload = {
                        "model": self.model_name,
                        "prompt": "Hello, how are you?",
                        "stream": False
                    }
                    
                    logger.info(f"Sending test request to {base_url}/api/generate")
                    response = await client.post(
                        f"{base_url}/api/generate",
                        json=test_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "response" in result:
                        logger.info(f"Ollama is available with model '{self.model_name}'")
                        logger.debug(f"Test response: {result['response'][:50]}...")
                        return True
                    else:
                        logger.warning(f"Unexpected response from Ollama API: {result}")
                        return False
                        
                except httpx.ConnectError as e:
                    logger.error(f"Connection error to Ollama: {str(e)}")
                    return False
                except Exception as e:
                    logger.error(f"Error checking Ollama availability: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in check_ollama_available: {str(e)}")
            return False
    
    def count_tokens(self, text: str) -> int:
        """
        Get a simple estimate of token count (more accurate than char counting).
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
            
        Note:
            This is a simplified token counting method that approximates the tokenization
            used by LLMs. For production use with specific models, consider using the
            model's actual tokenizer (e.g., tiktoken for OpenAI models).
            
            For Ollama/Llama models, this approximation is sufficient for context management.
        """
        # This is a simple approximation - words + punctuation
        # For more accuracy, we'd use a proper tokenizer library
        text = text.strip()
        # Count words (approximately 1 token each)
        words = len(re.findall(r'\b\w+\b', text))
        # Count punctuation and special characters (approximately 1 token each)
        special_chars = len(re.findall(r'[^\w\s]', text))
        # Estimate total tokens
        return words + special_chars
    
    async def retrieve_context(
        self,
        db: AsyncSession,
        user_id: int,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks to use as context for answering the query.
        
        This method performs vector similarity search to find the most relevant
        document chunks for the given query. It uses:
        
        1. Vector embeddings to represent both the query and document chunks
        2. Cosine similarity to measure relevance
        3. PostgreSQL with pgvector extension for efficient vector search
        
        The method can either:
        - Search across all user documents (if document_ids is None)
        - Search only within specific documents (if document_ids is provided)
        
        Args:
            db: Database session
            user_id: ID of the user
            query: User query
            document_ids: Optional list of specific document IDs to search within
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of document chunks with metadata and scores
        """
        logger.info(f"Retrieving context for query: '{query}' for user {user_id}")
        
        try:
            # If document_ids is specified, filter by those documents
            if document_ids and len(document_ids) > 0:
                # Generate embedding for query
                query_embedding = await DocumentService._generate_embeddings(query)
                
                # Construct a SQL query that filters by document_ids
                vector_search_query = """
                    WITH query_vector AS (
                        SELECT pgvector_parse_array(:embedding_json) AS vector
                    )
                    SELECT 
                        dc.id, 
                        dc.document_id, 
                        dc.chunk_index, 
                        dc.text, 
                        dc.embedding,
                        dc.chunk_metadata, 
                        dc.created_at,
                        d.filename,
                        1 - (dc.embedding <=> (SELECT vector FROM query_vector)) AS similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = :user_id
                    AND d.id = ANY(:document_ids)
                    ORDER BY dc.embedding <=> (SELECT vector FROM query_vector)
                    LIMIT :limit
                """
                
                # Create the pgvector_parse_array function if it doesn't exist
                setup_sql = """
                CREATE OR REPLACE FUNCTION pgvector_parse_array(arr text) RETURNS vector AS $$
                BEGIN
                    RETURN arr::vector;
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                """
                
                # Ensure helper function exists
                await db.execute(text(setup_sql))
                await db.commit()
                
                # Execute the query
                result = await db.execute(
                    text(vector_search_query), 
                    {
                        "user_id": user_id,
                        "document_ids": document_ids,  # Pass as a list, ANY() will handle it
                        "embedding_json": f"[{','.join(str(x) for x in query_embedding)}]",
                        "limit": top_k
                    }
                )
            else:
                # If no document_ids specified, use the standard search function
                chunks_with_scores = await DocumentService.search_documents(
                    db=db,
                    user_id=user_id,
                    query=query,
                    limit=top_k
                )
                
                # Format the results to match our expected structure
                context = []
                for chunk, score in chunks_with_scores:
                    # Fetch document info
                    document = await DocumentService.get_document_by_id(db, chunk.document_id)
                    if document:
                        context.append({
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "filename": document.filename,
                            "text": chunk.text,
                            "metadata": chunk.chunk_metadata,
                            "similarity_score": score
                        })
                return context
                
            # Process result rows
            context = []
            for row in result:
                context.append({
                    "chunk_id": row.id,
                    "document_id": row.document_id,
                    "filename": row.filename,
                    "text": row.text,
                    "metadata": row.chunk_metadata,
                    "similarity_score": float(row.similarity_score)
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved chunks into a context string for the LLM prompt.
        Also return citation information for source attribution.
        
        This method:
        1. Sorts chunks by relevance score (most relevant first)
        2. Formats each chunk with a citation marker [1], [2], etc.
        3. Truncates very long chunks to avoid context overflow
        4. Combines chunks into a single context string
        5. Checks total token count and removes least relevant chunks if needed
        6. Prepares citation metadata for the frontend
        
        Args:
            context_chunks: List of document chunks with metadata
            
        Returns:
            Tuple of (formatted_context, citations)
        """
        if not context_chunks:
            return "", []
        
        # Sort chunks by relevance score
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        formatted_chunks = []
        citations = []
        
        for i, chunk in enumerate(sorted_chunks):
            # Format each chunk with a citation marker
            citation_id = i + 1
            chunk_text = chunk.get("text", "").strip()
            
            # Clean and truncate the chunk if necessary
            if len(chunk_text) > 1000:  # Truncate very long chunks
                chunk_text = chunk_text[:1000] + "..."
            
            # Add to formatted chunks
            formatted_chunks.append(f"[{citation_id}] {chunk_text}")
            
            # Prepare citation information
            citations.append({
                "id": citation_id,
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename", "Unknown document"),
                "chunk_id": chunk.get("chunk_id"),
                "metadata": chunk.get("metadata", {})
            })
        
        # Combine all chunks with separators
        formatted_context = "\n\n".join(formatted_chunks)
        
        # Count tokens using our simple estimation method
        context_tokens = self.count_tokens(formatted_context)
        logger.info(f"Context contains approximately {context_tokens} tokens")
        
        # If context is too large, truncate it
        if context_tokens > self.max_tokens:
            logger.warning(f"Context too large ({context_tokens} tokens), truncating")
            
            # Truncate by removing chunks from the end until under token limit
            while context_tokens > self.max_tokens and formatted_chunks:
                formatted_chunks.pop()  # Remove the last (least relevant) chunk
                formatted_context = "\n\n".join(formatted_chunks)
                context_tokens = self.count_tokens(formatted_context)
                
                # Also remove the corresponding citation
                if citations:
                    citations.pop()
            
            logger.info(f"Truncated context to approximately {context_tokens} tokens")
        
        return formatted_context, citations
    
    def generate_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for the LLM with context and query.
        
        This creates a structured prompt that:
        1. Sets the role for the LLM (AI assistant answering questions)
        2. Provides the retrieved context information
        3. Presents the user's question
        4. Gives instructions on how to answer (use only context, cite sources, etc.)
        
        The prompt structure is optimized for Llama models but should work with
        most instruction-tuned LLMs.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Complete prompt string
        """
        # Create a prompt that includes context and instructs the model
        prompt = f"""You are an AI assistant answering questions based on the provided context information.
        
CONTEXT INFORMATION:
{context}

USER QUESTION: 
{query}

INSTRUCTIONS:
1. Answer the question using ONLY the provided context information
2. If the context doesn't contain enough information to answer the question fully, acknowledge the limitations
3. Use citation markers [1], [2], etc. to indicate which source you are referencing in your answer
4. IMPORTANT: Include citation numbers whenever you refer to information from a specific source
5. Be concise, accurate, and helpful
6. If you absolutely cannot answer the question based on the context, say "I don't have enough information to answer this question."

Your answer:
"""
        return prompt

    async def generate_answer_ollama(self, prompt: str) -> str:
        """
        Generate answer using Ollama API.
        
        This method sends the prompt to the Ollama API and retrieves the generated response.
        Ollama is a local LLM server that runs models like Llama on your own hardware.
        
        The method attempts to use LangChain if available, falling back to direct API calls.
        
        Args:
            prompt: The prompt to send to Ollama.
            
        Returns:
            The generated answer.
            
        Raises:
            Various exceptions for connection and API errors, which are caught and handled
            in the answer_question method.
        """
        logger.info("Generating answer using Ollama API")
        
        try:
            base_url = self.ollama_base_url.rstrip("/")
            
            # Try using LangChain if available
            if LANGCHAIN_AVAILABLE:
                try:
                    logger.info(f"Using LangChain with Ollama model: {self.model_name}")
                    ollama = OllamaLLM(
                        base_url=base_url,
                        model=self.model_name,
                        temperature=self.temperature
                    )
                    
                    # LangChain's invoke is synchronous, we should run it in a thread pool
                    # for proper async handling, but this is simpler for now
                    answer = ollama.invoke(prompt)
                    
                    if not answer or not isinstance(answer, str):
                        logger.error(f"Unexpected response from LangChain Ollama: {answer}")
                        raise ValueError("Empty or invalid response from LangChain Ollama")
                        
                    return answer.strip()
                    
                except Exception as e:
                    logger.warning(f"LangChain Ollama failed, falling back to direct API: {str(e)}")
                    # Fall through to direct API call
            
            # Direct API call - exactly matching the curl command that works
            url = f"{base_url}/api/generate"
            
            logger.info(f"Sending request to Ollama API with model: {self.model_name}")
            
            # Simplified payload matching exactly the curl command pattern
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Optional parameters - only add if needed
            if self.temperature != 0.7:  # Only add if not default
                payload["temperature"] = self.temperature
                
            if self.answer_max_tokens > 0:
                payload["num_predict"] = self.answer_max_tokens
            
            logger.debug(f"Ollama payload: {json.dumps(payload)}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    response = await client.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=payload
                    )
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    logger.debug(f"Ollama response: {json.dumps(result)[:200]}...")
                    
                    answer = result.get("response", "").strip()
                    
                    if not answer:
                        logger.error("Ollama API returned empty response")
                        return "Sorry, I encountered an error while generating the answer."
                    
                    return answer
                except httpx.ConnectError:
                    logger.error(f"Could not connect to Ollama at {url}. Is Ollama running?")
                    return "Could not connect to Ollama. Please make sure the Ollama server is running at the configured URL."
                except httpx.HTTPStatusError as e:
                    logger.error(f"Ollama API HTTP error: {e.response.status_code} - {e.response.text}")
                    
                    # Check for specific error codes
                    if e.response.status_code == 404:
                        return f"Model '{self.model_name}' not found. Please make sure you have pulled this model in Ollama."
                    elif e.response.status_code == 400:
                        return "Bad request to Ollama API. There might be an issue with the prompt or model configuration."
                    else:
                        return f"Ollama API error (HTTP {e.response.status_code}). Please check the server logs."
                except httpx.RequestError as e:
                    logger.error(f"Ollama API request error: {str(e)}")
                    return "Error connecting to Ollama API. Please check if Ollama is running correctly."
            
        except Exception as e:
            logger.error(f"Error using Ollama API: {str(e)}", exc_info=True)
            return "Sorry, I encountered an error while generating the answer."
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response if it exists and hasn't expired.
        
        Args:
            cache_key: Cache key for the query
            
        Returns:
            Cached response or None
        """
        if cache_key in QUERY_CACHE:
            entry = QUERY_CACHE[cache_key]
            # Check if cache entry is still valid
            if datetime.now() - entry["timestamp"] < CACHE_EXPIRY:
                logger.info(f"Cache hit for key: {cache_key}")
                return entry["data"]
            else:
                # Expired entry
                logger.info(f"Cache expired for key: {cache_key}")
                del QUERY_CACHE[cache_key]
        
        return None
    
    def store_cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Store a response in the cache with current timestamp.
        
        Args:
            cache_key: Cache key for the query
            response: Response data to cache
        """
        QUERY_CACHE[cache_key] = {
            "data": response,
            "timestamp": datetime.now()
        }
        logger.info(f"Stored response in cache with key: {cache_key}")
    
    def generate_cache_key(self, query: str, user_id: int, document_ids: Optional[List[int]] = None) -> str:
        """
        Generate a cache key for a specific query and document set.
        
        Args:
            query: User query
            user_id: User ID
            document_ids: Optional list of document IDs
            
        Returns:
            Cache key string
        """
        # Normalize and sort document_ids if provided
        doc_ids_str = ""
        if document_ids:
            doc_ids_str = "_".join(str(id) for id in sorted(document_ids))
            
        cache_key = f"u{user_id}_q{hash(query.lower().strip())}_d{doc_ids_str}"
        return cache_key
    
    async def answer_question(
        self, 
        db: AsyncSession,
        user_id: int,
        query: str,
        document_ids: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer to a question using retrieval-augmented generation.
        
        This is the main method that orchestrates the entire RAG pipeline:
        
        1. Check if Ollama is available
        2. Check cache for existing answers to the same query
        3. Retrieve relevant context chunks from documents
        4. Format context and prepare citations
        5. Generate a prompt combining context and query
        6. Send prompt to Ollama LLM to generate an answer
        7. Format and return the response with metadata
        8. Cache the response for future use
        
        The method includes extensive error handling and logging at each step.
        
        Args:
            db: Database session
            user_id: ID of the user
            query: The user's question
            document_ids: Optional list of document IDs to search within
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Log the incoming request for debugging
        logger.info(f"RAG query request - User: {user_id}, Query: '{query}', Document IDs: {document_ids}")
        
        start_time = datetime.now()
        
        try:
            # Check if Ollama is available
            ollama_available = await self.check_ollama_available()
            if not ollama_available:
                logger.error("Ollama is not available or the configured model doesn't exist")
                return {
                    "answer": f"The AI service is not available. Please ensure Ollama is running and the model '{self.model_name}' is installed.",
                    "citations": [],
                    "metadata": {
                        "processing_time": (datetime.now() - start_time).total_seconds(),
                        "query": query,
                        "document_ids": document_ids,
                        "context_chunks": 0,
                        "cached": False,
                        "model": self.model_name,
                        "provider": "Ollama",
                        "error": "Ollama service unavailable"
                    }
                }
            
            # Generate cache key based on query, user, and documents
            cache_key = self.generate_cache_key(query, user_id, document_ids)
            
            if use_cache:
                cached_response = self.get_cached_response(cache_key)
                if cached_response:
                    # Add metadata about cache
                    cached_response["metadata"]["cached"] = True
                    logger.info(f"Returning cached response for query: '{query}'")
                    return cached_response
            
            # Retrieve relevant context
            logger.info(f"Retrieving context for query: '{query}'")
            context_chunks = await self.retrieve_context(
                db=db,
                user_id=user_id,
                query=query,
                document_ids=document_ids,
                top_k=5  # Retrieve top 5 most relevant chunks
            )
            
            # Check if there's no context retrieved
            if not context_chunks:
                logger.warning(f"No context retrieved for query: '{query}'")
                return {
                    "answer": "I don't have enough information to answer this question based on the available documents.",
                    "citations": [],
                    "metadata": {
                        "processing_time": (datetime.now() - start_time).total_seconds(),
                        "query": query,
                        "document_ids": document_ids,
                        "context_chunks": 0,
                        "cached": False,
                        "model": self.model_name,
                        "provider": "Ollama"
                    }
                }
            
            # Format context for prompt
            context_text, citations = self.format_context(context_chunks)
            
            # Generate prompt with context and query
            prompt = self.generate_prompt(query, context_text)
            
            # Generate answer using Ollama
            error_message = None
            try:
                logger.info(f"Generating answer using Ollama with model {self.model_name}")
                answer = await self.generate_answer_ollama(prompt)
            except Exception as e:
                error_class = e.__class__.__name__
                logger.error(f"Ollama LLM failed with {error_class}: {str(e)}")
                error_message = f"LLM service unavailable: {str(e)}"
                
                if "connection" in str(e).lower():
                    answer = "Could not connect to the local Ollama server. Please ensure Ollama is running on your system."
                else:
                    answer = "Sorry, the AI service is currently unavailable. Please try again later."
            
            # Check if we got an error message back
            if answer and answer.startswith("Sorry, I encountered an error"):
                error_message = "Error generating answer from LLM"
                logger.error(f"Error response from LLM for query: '{query}'")
            else:
                logger.info(f"Successfully generated answer for query: '{query}'")
            
            # Prepare response
            response = {
                "answer": answer if answer else "Sorry, I was unable to generate an answer at this time.",
                "citations": citations,
                "metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "query": query,
                    "document_ids": document_ids,
                    "context_chunks": len(context_chunks),
                    "cached": False,
                    "model": self.model_name,
                    "provider": "Ollama"
                }
            }
            
            # Add error information if applicable
            if error_message:
                response["metadata"]["error"] = error_message
            
            # Cache the response
            if use_cache and not error_message:
                self.store_cache_response(self.generate_cache_key(query, user_id, document_ids), response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}", exc_info=True)
            
            # Create error response
            error_response = {
                "answer": "Sorry, I encountered an error while processing your question.",
                "citations": [],
                "metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "query": query,
                    "document_ids": document_ids,
                    "context_chunks": 0,
                    "cached": False,
                    "error": str(e)
                }
            }
            
            return error_response
    
    async def get_available_documents(
        self,
        db: AsyncSession,
        user_id: int
    ) -> List[Dict[str, Any]]:
        """
        DEPRECATED: This method is kept for backward compatibility with tests.
        Use DocumentService.get_documents_by_user_id() instead and filter for processed documents.
        
        Get a list of available documents for the user that can be used for Q&A.
        
        Args:
            db: Database session
            user_id: ID of the user
            
        Returns:
            List of documents with id, filename, and content_type
        """
        try:
            import warnings
            warnings.warn(
                "RAGService.get_available_documents is deprecated. "
                "Use DocumentService.get_documents_by_user_id instead.",
                DeprecationWarning, stacklevel=2
            )
            
            documents, _ = await DocumentService.get_documents_by_user_id(
                db=db,
                user_id=user_id,
                skip=0,
                limit=100  # Reasonable limit for document selection
            )
            
            available_docs = []
            for doc in documents:
                # Only include processed documents
                if doc.status == "processed":
                    available_docs.append({
                        "id": doc.id,
                        "filename": doc.filename,
                        "content_type": doc.content_type,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None
                    })
                
            return available_docs
            
        except Exception as e:
            logger.error(f"Error getting available documents: {str(e)}")
            return []


# Create a singleton instance
rag_service = RAGService() 