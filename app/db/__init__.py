"""
Database package for FastAPI User Management System.

This package contains modules for database connectivity, ORM setup, and session management:

- base_class.py: Defines the SQLAlchemy declarative Base class
- model_registry.py: Registers all models with SQLAlchemy for migration support
- session.py: Handles database connection and session management

The package follows a structure designed to prevent circular imports while
providing clean integration with SQLAlchemy and Alembic.
""" 