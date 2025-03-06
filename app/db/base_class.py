"""
SQLAlchemy Base class definition.

This module defines the SQLAlchemy declarative base class that all ORM models
will inherit from. It provides the foundation for SQLAlchemy's ORM functionality.

Having this in a separate file prevents circular imports, as models can import
Base from here, while model_registry.py can import both Base and all models.
"""

from sqlalchemy.orm import declarative_base

# Create a base class for SQLAlchemy models
Base = declarative_base() 