# Embedding Generation and Storage Optimization

This document outlines the optimizations made to the embedding generation and storage process in the application.

## Changes Made

1. **Batch Embedding Generation**
   - Added a new method `_generate_embeddings_batch` to process multiple text chunks in a single batch
   - This significantly improves performance by reducing the number of calls to the embedding model
   - The batch method handles invalid inputs gracefully, returning zero vectors for invalid texts

2. **Async Database Support**
   - Added proper async SQLAlchemy session handling with `AsyncSession`
   - Created `get_async_db()` dependency for FastAPI routes
   - Updated database URL configuration to support async drivers (asyncpg for PostgreSQL, aiosqlite for SQLite)

3. **Document Processing Optimization**
   - Updated the document processing method to use batch embedding generation
   - Improved error handling and logging during document processing
   - Added validation to ensure only valid text chunks are processed

4. **API Dependencies**
   - Added async versions of authentication dependencies
   - Created async versions of user authentication methods

## Performance Improvements

The batch embedding generation provides several performance benefits:

1. **Reduced Model Loading Overhead**: The embedding model is loaded only once for multiple chunks
2. **Vectorized Operations**: The SentenceTransformer model can process multiple texts more efficiently in a batch
3. **Reduced Event Loop Blocking**: By processing embeddings in a batch, we reduce the number of times the event loop is blocked

## Database Compatibility

The application now supports both synchronous and asynchronous database operations:

- **Synchronous**: Uses standard SQLAlchemy with psycopg2 (PostgreSQL) or sqlite3 (SQLite)
- **Asynchronous**: Uses SQLAlchemy async with asyncpg (PostgreSQL) or aiosqlite (SQLite)

## How to Use

1. Install the required dependencies:
   ```bash
   ./install_async_deps.sh
   ```

2. Run the application with async database support:
   ```bash
   python run.py
   ```

## Testing

New tests have been added to verify the batch embedding generation:

- `test_generate_embeddings_batch`: Tests basic batch embedding generation
- `test_generate_embeddings_batch_with_invalid_inputs`: Tests handling of invalid inputs
- `test_generate_embeddings_batch_empty_list`: Tests handling of empty input lists

Run the tests with:
```bash
python -m pytest tests/test_services/test_batch_embeddings.py -v
``` 