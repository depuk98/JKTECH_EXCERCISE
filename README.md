# FastAPI Document Processing System

A FastAPI-based document processing system with vector embeddings for semantic search and document management.

## Features

- User authentication and management
- Document upload and processing (PDF, DOCX, TXT)
- Hierarchical text chunking with sliding window 
- Vector embeddings generation using Sentence Transformers
- Vector storage with PostgreSQL + pgvector
- Semantic search across document contents
- Modern UI with Bootstrap
- Background processing of documents
- Comprehensive test coverage

## Setup Instructions

### Prerequisites

1. Python 3.9+
2. PostgreSQL with pgvector extension
3. Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fastapi-document-processing
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the pgvector extension for PostgreSQL:
   ```bash
   psql -U your_username -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

5. Create a `.env` file with environment variables:
   ```
   DATABASE_URL=postgresql+asyncpg://username:password@localhost/dbname
   POSTGRES_USER=username
   POSTGRES_PASSWORD=password
   POSTGRES_DB=dbname
   SECRET_KEY=your_secret_key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=1440
   ```

6. Run migrations:
   ```bash
   alembic upgrade head
   ```

### Running the Application

1. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Access the application at `http://localhost:8000`
3. API documentation is available at `http://localhost:8000/docs`

## Usage

### Document Processing

1. Log in to the application
2. Navigate to the Documents page
3. Upload a document (PDF, DOCX, or TXT)
4. The system will process the document in the background:
   - Extract text content
   - Split into chunks with logical boundaries
   - Generate vector embeddings
   - Store in the database

### Search

1. Use the search bar to perform semantic searches across your documents
2. Results are ranked by relevance based on vector similarity

## Architecture

The application follows a clean, modular architecture:

- **Models**: SQLAlchemy models for database tables
- **Schemas**: Pydantic schemas for validation and serialization
- **Services**: Business logic layer
- **API Routes**: HTTP endpoints
- **Core**: Configuration and dependencies

## Database Schema

- **users**: User information and authentication
- **documents**: Document metadata
- **document_chunks**: Document content with vector embeddings

## Vector Search

Document search uses the pgvector extension to perform efficient similarity searches:

1. The query text is converted to an embedding vector
2. A vector similarity search is performed (using cosine similarity)
3. Results are returned sorted by similarity score

## Testing

Run the test suite with:

```bash
pytest
```

## License

MIT 