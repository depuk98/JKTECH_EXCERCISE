"""Database initialization utilities.

This module provides functions to initialize the database tables for both
synchronous and asynchronous database engines.
"""

import logging
import asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import Engine, inspect

from app.db.base_class import Base

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db_sync(engine: Engine) -> None:
    """
    Initialize database tables using synchronous engine.
    Only creates tables if they don't already exist.
    
    Args:
        engine: SQLAlchemy engine
    """
    # Check if tables already exist
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    if "users" in existing_tables and "documents" in existing_tables:
        logger.info("Database tables already exist, skipping initialization")
        return
    
    # Tables don't exist, create them
    logger.info("Creating database tables with synchronous engine")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

async def init_db_async(async_engine: AsyncEngine) -> None:
    """
    Initialize database tables using asynchronous engine.
    Only creates tables if they don't already exist.
    
    Args:
        async_engine: SQLAlchemy async engine
    """
    # Check if tables already exist
    async with async_engine.connect() as conn:
        # Get the inspector
        inspector = await conn.run_sync(inspect)
        
        # Check for essential tables
        existing_tables = await conn.run_sync(lambda sync_conn: inspector.get_table_names())
        
        if "users" in existing_tables and "documents" in existing_tables:
            logger.info("Database tables already exist, skipping initialization")
            return
    
    # Tables don't exist, create them
    logger.info("Creating database tables with asynchronous engine")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully") 