import pytest
from sqlalchemy import inspect
import importlib

from app.db.base_class import Base
from app.db.model_registry import *  # This intentionally imports everything from model_registry
from app.models.user import User
from app.models.document import Document, DocumentChunk

def test_model_registry_contains_all_models():
    """
    Test that all models are properly imported and registered with Base.metadata.
    This test ensures that the model_registry is correctly importing all models
    so that Alembic can detect them for migrations.
    """
    # Get all tables registered with Base.metadata
    metadata = Base.metadata
    
    # Check if metadata has tables (in memory)
    tables = metadata.tables
    assert tables, "No tables found in Base.metadata"
    
    # Check that core models are registered
    user_table = tables.get('users')
    assert user_table is not None, "User model not registered with Base.metadata"
    
    document_table = tables.get('documents')
    assert document_table is not None, "Document model not registered with Base.metadata"
    
    document_chunk_table = tables.get('document_chunks')
    assert document_chunk_table is not None, "DocumentChunk model not registered with Base.metadata"
    
    # Check that key columns exist in the models
    user_columns = [c.name for c in user_table.columns]
    assert 'id' in user_columns
    assert 'email' in user_columns
    assert 'hashed_password' in user_columns
    
    document_columns = [c.name for c in document_table.columns]
    assert 'id' in document_columns
    assert 'filename' in document_columns
    assert 'user_id' in document_columns
    
    chunk_columns = [c.name for c in document_chunk_table.columns]
    assert 'id' in chunk_columns
    assert 'document_id' in chunk_columns
    assert 'text' in chunk_columns  # Note: might be 'text' instead of 'content'
    
    # This test ensures that any new models added to the application
    # are also imported in model_registry.py for Alembic to detect them

def test_model_registry_import_mechanism():
    """
    Test the import mechanism of the model registry.
    Ensures that the module can be imported and reloaded properly.
    """
    # Import the module
    model_registry = importlib.import_module('app.db.model_registry')
    
    # Reload the module to ensure it can be reloaded without errors
    importlib.reload(model_registry)
    
    # Check that the module exports the expected attributes
    assert hasattr(model_registry, '__all__'), "model_registry should have __all__ attribute"
    
    # Check for imported models in module globals
    module_globals = model_registry.__dict__
    assert 'User' in module_globals, "User model should be imported in model_registry"
    assert 'Document' in module_globals, "Document model should be imported in model_registry"
    assert 'DocumentChunk' in module_globals, "DocumentChunk model should be imported in model_registry"

def test_model_registry_base_integration():
    """
    Test that the Base class properly includes all models from model_registry.
    """
    # Import Base directly
    from app.db.base_class import Base
    
    # Check that Base includes models from model_registry
    all_tables = Base.metadata.tables
    
    # Make sure we have the expected tables in Base
    expected_tables = ['users', 'documents', 'document_chunks']
    for table_name in expected_tables:
        assert table_name in all_tables, f"Table {table_name} should be in Base.metadata.tables"
    
    # Check table relationships
    document_table = all_tables['documents']
    document_chunk_table = all_tables['document_chunks']
    
    # Check foreign key relationship between documents and chunks
    found_fk = False
    for fk in document_chunk_table.foreign_keys:
        if fk.column.table == document_table:
            found_fk = True
            break
    
    assert found_fk, "Foreign key relationship between documents and chunks should exist" 