# FastAPI Document Processing System

A FastAPI-based document processing system with vector embeddings for semantic search, document management, and question answering capabilities.

## Features

- User authentication and management
- Document upload and processing (PDF, DOCX, TXT)
- Hierarchical text chunking with sliding window 
- Vector embeddings generation using Sentence Transformers
- Vector storage with PostgreSQL + pgvector
- Semantic search across document contents
- **Document-based Question Answering (Q&A)**
- **Integration with OpenAI and Ollama LLMs**
- Modern UI with Bootstrap
- Background processing of documents
- Comprehensive test coverage

## Setup Instructions

### Prerequisites

1. Python 3.9+
2. PostgreSQL with pgvector extension
3. Virtual environment (recommended)
4. [Optional] Ollama for local LLM support

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
   OPENAI_API_KEY=your_openai_api_key  # Optional, for OpenAI integration
   OLLAMA_BASE_URL=http://localhost:11434  # Optional, for Ollama integration
   LLM_MODEL=llama3.1  # Options: gpt-4, gpt-3.5-turbo, llama3.1
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

### Question Answering (Q&A)

1. Navigate to the Q&A page
2. Select the documents you want to query or choose "All Documents"
3. Enter your question in the input field
4. The system will:
   - Retrieve relevant document chunks using vector similarity
   - Generate an answer based on the retrieved context
   - Provide citations to source documents
5. View the answer with references to the original documents

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

## Vector Search and RAG

The application uses a Retrieval-Augmented Generation (RAG) approach:

1. Document search uses the pgvector extension to perform efficient similarity searches:
   - The query text is converted to an embedding vector
   - A vector similarity search is performed (using cosine similarity)
   - Results are returned sorted by similarity score

2. For Q&A, the system:
   - Retrieves relevant document chunks using vector search
   - Formats the chunks with citations as context
   - Passes the context to an LLM (OpenAI or Ollama)
   - Returns the generated answer with citations

## Testing

Run the test suite with:

```bash
pytest
```

For coverage reports:

```bash
pytest --cov=app
```

## License

MIT 