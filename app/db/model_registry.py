"""
Model registry module.

This module imports all SQLAlchemy models and makes them available to Alembic
for auto-generating migrations. Any new models should be imported here to ensure
they are included in migrations.

This file serves as a central registry to solve the SQLAlchemy/Alembic circular import problem:
1. Models import Base from base_class.py
2. This file imports all models and Base
3. Alembic imports this file to access all models through Base.metadata
"""

from app.db.base_class import Base

# Import all models here so they are registered with SQLAlchemy Base.metadata
from app.models.user import User
from app.models.document import Document, DocumentChunk

# Add future models imports here
# from app.models.item import Item
# from app.models.role import Role
# etc. 