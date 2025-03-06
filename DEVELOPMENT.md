# JKT Application - Development Guide

This guide provides comprehensive information about the JKT application architecture, implementation details, and development practices.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Database Design](#database-design)
4. [API Design](#api-design)
5. [Authentication & Authorization](#authentication--authorization)
6. [Core Workflows](#core-workflows)
7. [Technology Stack](#technology-stack)
8. [Code Organization](#code-organization)
9. [Implementation Details](#implementation-details)
10. [Asynchronous Programming](#asynchronous-programming)
11. [Performance Optimization](#performance-optimization)
12. [Security Considerations](#security-considerations)
13. [Deployment](#deployment)
14. [Development Workflow](#development-workflow)
15. [Troubleshooting](#troubleshooting)

## Architecture Overview

The JKT application follows a modern, layered architecture pattern with clear separation of concerns:

```
               ┌───────────────┐
               │   Client UI   │
               └───────┬───────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│              FastAPI App                │
├─────────────────────────────────────────┤
│ ┌─────────────┐  ┌────────────────────┐ │
│ │ API Routes  │  │  Dependency Inj.   │ │
│ └──────┬──────┘  └────────┬───────────┘ │
│        │                  │             │
│ ┌──────▼──────┐  ┌────────▼───────────┐ │
│ │  Services   │  │  Auth & Security   │ │
│ └──────┬──────┘  └────────────────────┘ │
│        │                                │
│ ┌──────▼───────────────────────────────┐│
│ │           Data Access Layer          ││
│ └──────────────────┬───────────────────┘│
└────────────────────┼────────────────────┘
                     │
                     ▼
             ┌───────────────┐
             │   Database    │
             └───────────────┘
```

The architecture follows these key principles:
- **Layered Design**: Clean separation between API layer, business logic, and data access
- **Dependency Injection**: FastAPI's dependency system for flexible component composition
- **RESTful APIs**: Well-defined resource-oriented endpoints
- **Asynchronous Operations**: Utilizes Python's async capabilities for improved performance

## System Components

### 1. API Layer (`app/api/`)
- **Endpoints**: RESTful API endpoints organized by resource
- **Request Validation**: Uses Pydantic models to validate incoming requests
- **Response Formatting**: Consistent response structures and status codes
- **Error Handling**: Unified error handling and reporting

### 2. Service Layer (`app/services/`)
- **Business Logic**: Encapsulates core application logic
- **Orchestration**: Coordinates operations across multiple resources
- **Transaction Management**: Ensures data consistency
- **Event Handling**: Manages application events and side effects

### 3. Data Access Layer (`app/db/`, `app/models/`)
- **ORM Models**: SQLAlchemy models representing database entities
- **Query Construction**: Database query building and execution
- **Connection Management**: Database connection pooling and lifecycle
- **Transaction Control**: Database transaction boundaries

### 4. Schema Layer (`app/schemas/`)
- **Data Validation**: Pydantic models for request/response validation
- **Data Transformation**: Converting between API and domain models
- **Documentation**: Self-documenting API schemas

### 5. Core Infrastructure (`app/core/`)
- **Configuration**: Application settings and environment
- **Security**: Authentication, authorization, and security policies
- **Logging**: Application logging and monitoring
- **Exception Handling**: Global exception management

### 6. Static Resources (`app/static/`, `app/templates/`)
- **UI Assets**: Static files for the web interface
- **Templates**: Server-side rendering templates

## Database Design

### Entity Relationship Diagram

```
┌─────────────┐       ┌───────────────┐       ┌─────────────────┐
│   User      │       │   Document    │       │ Document_Chunk  │
├─────────────┤       ├───────────────┤       ├─────────────────┤
│ id          │       │ id            │       │ id              │
│ email       │       │ user_id       │◄──┐   │ document_id     │◄─┐
│ username    │       │ filename      │   │   │ chunk_index     │  │
│ password    │┌─────►│ content_type  │   └───│ text            │  │
│ is_active   ││      │ file_path     │       │ embedding       │  │
│ is_superuser││      │ file_size     │       │ chunk_metadata  │  │
│ created_at  ││      │ page_count    │       │ created_at      │  │
│ updated_at  ││      │ status        │       └─────────────────┘  │
│ age         ││      │ error_message │                            │
└─────────────┘│      │ created_at    │                            │
               │      │ updated_at    │                            │
               └──────┤ user_id       │                            │
                      └───────────────┘                            │
                              ▲                                    │
                              └────────────────────────────────────┘
```

### Key Tables

1. **User**
   - Primary entity for authentication and authorization
   - Stores user credentials and profile information
   - Linked to documents via one-to-many relationship

2. **Document**
   - Represents uploaded files (PDFs, text files, etc.)
   - Contains metadata about the document
   - Processing status tracking
   - Belongs to a user

3. **Document_Chunk**
   - Represents segments of a document processed for AI operations
   - Contains embeddings for vector search
   - Contains the actual text content of chunks
   - Parent-child relationship with Document

## API Design

The API follows RESTful principles, with resources organized around business entities and consistent naming conventions.

### Auth API Endpoints

```
POST   /api/auth/login          - Authenticate user and get token
POST   /api/auth/refresh-token  - Refresh access token
POST   /api/auth/reset-password - Request password reset
```

### User API Endpoints

```
GET    /api/users/me            - Get current user profile
PUT    /api/users/me            - Update current user profile
GET    /api/users/{id}          - Get user by ID (admin only)
POST   /api/users/              - Create new user
PUT    /api/users/{id}          - Update user (admin only)
DELETE /api/users/{id}          - Delete user (admin only)
```

### Document API Endpoints

```
GET    /api/documents/          - List documents (with pagination)
POST   /api/documents/          - Upload new document
GET    /api/documents/{id}      - Get document details
PUT    /api/documents/{id}      - Update document metadata
DELETE /api/documents/{id}      - Delete document
```

### Search API Endpoints

```
GET    /api/search/             - Search across documents
```

## Authentication & Authorization

### Authentication Flow

1. **Login Request**: Client submits credentials
2. **Credential Verification**: Server validates credentials against database
3. **Token Generation**: JWT token generated with user claims
4. **Token Response**: Token returned to client
5. **Authenticated Requests**: Client includes token in Authorization header

### JWT Structure

```
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_id",
    "exp": 1672531200,
    "iat": 1672527600,
    "jti": "unique_token_id",
    "type": "access",
    "is_superuser": false
  },
  "signature": "..."
}
```

### Authorization

Access control is implemented using:
- **Role-based access control**: Different permissions for normal users vs. superusers
- **Resource ownership**: Users can only access their own resources by default
- **Permission checking**: Dependencies that verify user permissions before endpoint execution

## Technology Stack

### Backend
- **Framework**: FastAPI
- **ORM**: SQLAlchemy
- **Database**: PostgreSQL, SQLite
- **Authentication**: JWT with Passlib + Bcrypt
- **Task Processing**: Asynchronous with asyncio
- **API Documentation**: OpenAPI (Swagger UI + ReDoc)

### Data Processing
- **Vector Embeddings**: Sentence Transformers
- **Vector Storage**: PostgreSQL with pgvector extension
- **Document Processing**: PyPDF2, python-docx, etc.

### DevOps
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Configuration**: Environment variables with dotenv
- **Database Migrations**: Alembic

## Code Organization

```
JKT_EX/
├── alembic/                # Database migrations
│   ├── versions/           # Migration version files
│   └── env.py              # Alembic environment configuration
├── app/                    # Main application
│   ├── api/                # API endpoints
│   │   ├── deps.py         # API dependencies
│   │   ├── api.py          # API router
│   │   └── endpoints/      # API endpoint modules
│   │       ├── auth.py     # Authentication endpoints
│   │       ├── users.py    # User endpoints
│   │       └── documents.py # Document endpoints
│   ├── core/               # Core application components
│   │   ├── config.py       # Configuration settings
│   │   ├── security.py     # Security utilities
│   │   └── exceptions.py   # Custom exceptions
│   ├── db/                 # Database components
│   │   ├── base_class.py   # SQLAlchemy base class
│   │   ├── session.py      # Database session management
│   │   └── init_db.py      # Database initialization
│   ├── models/             # SQLAlchemy models
│   │   ├── user.py         # User model
│   │   └── document.py     # Document model
│   ├── schemas/            # Pydantic schemas
│   │   ├── user.py         # User schemas
│   │   └── document.py     # Document schemas
│   ├── services/           # Business logic services
│   │   ├── user.py         # User service
│   │   └── document.py     # Document service
│   ├── utils/              # Utility functions
│   │   ├── file.py         # File handling utilities
│   │   └── embeddings.py   # Text embedding utilities
│   ├── templates/          # HTML templates
│   ├── static/             # Static resources
│   ├── main.py             # Application entry point
│   └── __init__.py         # Package initialization
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures
│   ├── test_api/           # API tests
│   ├── test_db/            # Database tests
│   ├── test_integration/   # Integration tests
│   ├── test_performance/   # Performance tests
│   └── test_services/      # Service tests
├── .env                    # Environment variables
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── alembic.ini             # Alembic configuration
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Implementation Details

### FastAPI Application Setup

```python
# app/main.py
from fastapi import FastAPI
from app.api.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.exceptions import CustomException

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message},
    )
```

### Database Configuration

```python
# app/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from app.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async engine for PostgreSQL
if 'postgresql' in settings.DATABASE_URL:
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
        future=True,
    )
    AsyncSessionLocal = sessionmaker(
        bind=async_engine, expire_on_commit=False, class_=AsyncSession
    )
```

### Dependency Injection

```python
# app/api/deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.user import User
from app.schemas.token import TokenPayload

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

def get_db():
    """
    Dependency to get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    """
    Dependency to get the current authenticated user.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = db.query(User).get(token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user
```

### Data Models

```python
# app/models/user.py
from sqlalchemy import Boolean, Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base_class import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    age = Column(Integer, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
```

### Service Layer

```python
# app/services/document.py
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
import os
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentUpdate
from app.core.config import settings
from app.utils.file import save_upload_file, get_file_size

class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        
    def get(self, id: int) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == id).first()
    
    def get_multi_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        return self.db.query(Document).filter(
            Document.user_id == user_id
        ).offset(skip).limit(limit).all()
    
    def create(self, obj_in: DocumentCreate, file: UploadFile, user_id: int) -> Document:
        # Save file to disk
        file_path = save_upload_file(file)
        file_size = get_file_size(file_path)
        
        # Create document in DB
        db_obj = Document(
            user_id=user_id,
            filename=obj_in.filename or file.filename,
            content_type=file.content_type,
            file_path=file_path,
            file_size=file_size,
            status="pending",
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        # Return the document
        return db_obj
```

## Asynchronous Programming

The application uses Python's async/await pattern for improved performance, especially for I/O-bound operations.

### Async Database Operations

```python
# app/services/async_document.py
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import Document

class AsyncDocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def get(self, id: int) -> Optional[Document]:
        result = await self.db.execute(
            select(Document).where(Document.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_multi_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        result = await self.db.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
```

### Async Session Management

When working with async SQLAlchemy sessions, always use one of these patterns:

#### 1. Async Context Manager (Recommended)

```python
async def process_data():
    async with AsyncSessionLocal() as session:
        # Do work with session
        # Session is automatically closed correctly
```

#### 2. Try/Finally Pattern

```python
async def process_data():
    session = AsyncSessionLocal()
    try:
        # Do work with session
        return result
    finally:
        await session.close()  # Must await the close!
```

### Common Async Pitfalls

#### "Coroutine was never awaited" Warning

This warning occurs when an async function is called but not awaited:

```python
# WRONG - will cause warning
session = AsyncSessionLocal()
session.close()  # This is a coroutine that should be awaited

# CORRECT
session = AsyncSessionLocal()
await session.close()
```

#### Mixing Sync and Async Code

```python
# WRONG - calling async method from sync code
def some_function():
    async_db = AsyncSessionLocal()
    async_db.close()  # This will cause a warning

# CORRECT - use only sync methods in sync functions
def some_function():
    db = SessionLocal()  # Use sync session
    db.close()
```

## Performance Optimization

### Database Query Optimization

```python
# Efficient querying with specific column selection
def get_user_email(user_id: int, db: Session) -> str:
    result = db.query(User.email).filter(User.id == user_id).first()
    return result[0] if result else None

# Using SQLAlchemy execution plans
from sqlalchemy import select
stmt = select(User).where(User.id == user_id)
plan = db.execute(statement=stmt)
```

### Asynchronous Data Fetching

```python
# Asynchronous data fetching patterns
async def get_user_with_documents(user_id: int, async_db: AsyncSession):
    # Execute queries in parallel
    user_task = asyncio.create_task(
        async_db.execute(select(User).where(User.id == user_id))
    )
    
    docs_task = asyncio.create_task(
        async_db.execute(
            select(Document).where(Document.user_id == user_id)
        )
    )
    
    # Await both results
    user_result, docs_result = await asyncio.gather(user_task, docs_task)
    
    user = user_result.scalar_one_or_none()
    docs = docs_result.scalars().all()
    
    return {"user": user, "documents": docs}
```

### Connection Pooling

```python
# Connection pooling configuration
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=20,  # Maximum number of connections
    max_overflow=30,  # Allow temporarily exceeding pool_size
    pool_timeout=30,  # Seconds to wait for a connection
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Verify connections before using
)
```

## Security Considerations

### Authentication Security
- **Password Storage**: Bcrypt hashing with appropriate work factor
- **JWT Security**: Short-lived tokens, secure token handling
- **CSRF Protection**: Token-based CSRF protection

### API Security
- **Input Validation**: All endpoints validate input with Pydantic
- **Rate Limiting**: Prevent abuse through request rate limiting
- **CORS Policy**: Strict cross-origin resource sharing policy

### Data Security
- **Data Encryption**: Sensitive data encrypted at rest
- **SQL Injection Prevention**: Parameterized queries via ORM
- **File Upload Security**: Content-type validation, size limits

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8000

# Run with gunicorn and uvicorn workers
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - db

  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    env_file:
      - .env
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Production Considerations

1. **Environment-specific configurations**:
   ```
   APP_ENV=production
   LOG_LEVEL=ERROR
   DATABASE_URL=postgresql://user:password@db:5432/prod_db
   ```

2. **Health check endpoints**:
   ```python
   @app.get("/health")
   async def health_check():
       return {"status": "ok"}
   ```

3. **Monitoring and logging**:
   ```python
   import logging
   from app.core.config import settings
   
   logging.basicConfig(
       level=getattr(logging, settings.LOG_LEVEL),
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
   )
   
   logger = logging.getLogger(__name__)
   ```

## Development Workflow

### Setting Up the Development Environment

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Set up environment variables
5. Run database migrations
6. Start the development server

### Making Changes

1. Create a feature branch
2. Make changes to the code
3. Run tests to ensure functionality
4. Commit changes with descriptive messages
5. Push changes and create a pull request

### Code Style and Standards

- Follow PEP 8 for Python code style
- Use type hints for function parameters and return values
- Write docstrings for all functions, classes, and modules
- Keep functions small and focused on a single responsibility
- Use meaningful variable and function names

## Troubleshooting

### Async Session Warnings

If you see "coroutine was never awaited" warnings:

1. Ensure all async functions are properly awaited
2. Use async context managers for session management
3. Don't mix sync and async code inappropriately
4. Check test fixtures for proper async session handling

### Database Connection Issues

1. Verify database URL in environment variables
2. Check database server is running
3. Ensure database user has appropriate permissions
4. Look for connection pool exhaustion in logs

### Performance Problems

1. Use database indexing for frequently queried fields
2. Implement pagination for list endpoints
3. Use async operations for I/O-bound tasks
4. Consider caching for frequently accessed data
5. Profile the application to identify bottlenecks 