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

from app.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL, 
    # connect_args={"check_same_thread": False}
)

# Create session factory
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