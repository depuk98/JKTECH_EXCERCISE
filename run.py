import os
import asyncio
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Update DATABASE_URL for async drivers if needed
def update_database_url():
    """Update DATABASE_URL to use async drivers."""
    db_url = os.environ.get("DATABASE_URL", "")
    
    if "postgresql" in db_url and "postgresql+asyncpg" not in db_url:
        os.environ["DATABASE_URL"] = db_url.replace("postgresql://", "postgresql+asyncpg://")
        print(f"Updated DATABASE_URL to use asyncpg driver: {os.environ['DATABASE_URL']}")
    elif "sqlite" in db_url and "sqlite+aiosqlite" not in db_url:
        os.environ["DATABASE_URL"] = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        print(f"Updated DATABASE_URL to use aiosqlite driver: {os.environ['DATABASE_URL']}")
    else:
        print(f"Using DATABASE_URL as is: {db_url}")

# Initialize database with async engine
async def init_async_db():
    """Initialize database with async engine."""
    from app.db.session import async_engine
    from app.db.init_db import init_db_async
    
    try:
        await init_db_async(async_engine)
        print("Async database initialization complete")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

async def main():
    """Main entry point for async initialization."""
    # Update DATABASE_URL for async drivers
    update_database_url()
    
    # Initialize database
    await init_async_db()
    
    # Import app after database initialization to avoid circular imports
    from app.main import app
    
    # Run the application
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=True)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 