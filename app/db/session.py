"""
Database session management.

This module provides components for connecting to the database and managing 
database sessions throughout the application lifecycle.

It defines:
- The SQLAlchemy engine (connection to the database)
- The session factory for creating new database sessions
- A FastAPI dependency function for database session management in API endpoints
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from app.core.config import settings

# Create SQLAlchemy engine for synchronous operations
engine = create_engine(
    settings.DATABASE_URL, 
    # connect_args={"check_same_thread": False}
)

# Create session factory for synchronous operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a function to get a database session
def get_db():
    """
    FastAPI dependency that provides a SQLAlchemy database session.
    
    Yields a database session that is automatically closed when the request is finished.
    This ensures proper resource management and connection pooling.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create async engine and session factory
# For PostgreSQL, replace postgresql:// with postgresql+asyncpg://
# For SQLite, replace sqlite:// with sqlite+aiosqlite://
if 'postgresql' in settings.DATABASE_URL:
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
    )
elif 'sqlite' in settings.DATABASE_URL:
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace('sqlite:///', 'sqlite+aiosqlite:///'),
    )
else:
    # Fallback to the original URL if it's not PostgreSQL or SQLite
    async_engine = create_async_engine(settings.DATABASE_URL)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    bind=async_engine
)

# Create a function to get an async database session
async def get_async_db():
    """
    FastAPI dependency that provides an async SQLAlchemy database session.
    
    Yields an async database session that is automatically closed when the request is finished.
    
    Yields:
        AsyncSession: Async SQLAlchemy database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 