import os
import pytest
from typing import Dict, Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
# Import Base directly from base_class for test database initialization
from app.db.base_class import Base
from app.db.session import get_db
from app.core.config import settings
from app.services.user import UserService
from app.schemas.user import UserCreate

# Create a test database
TEST_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Create test session
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db() -> Generator:
    """
    Create a fresh database for each test.
    """
    # Create the database tables
    Base.metadata.create_all(bind=engine)
    
    # Create a new session for testing
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop the database tables
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db) -> Generator:
    """
    Create a test client with the test database.
    """
    # Override the get_db dependency
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    # Override the dependency
    app.dependency_overrides[get_db] = override_get_db
    
    # Create a test client
    with TestClient(app) as c:
        yield c
    
    # Clear the dependency override
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def normal_user(db) -> Dict[str, str]:
    """
    Create a normal user for testing.
    """
    user_in = UserCreate(
        email="user@example.com",
        username="testuser",
        password="password123",
    )
    user = UserService.create(db, user_in=user_in)
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "password": "password123",
    }

@pytest.fixture(scope="function")
def superuser(db) -> Dict[str, str]:
    """
    Create a superuser for testing.
    """
    user_in = UserCreate(
        email="admin@example.com",
        username="admin",
        password="adminpass123",
        is_superuser=True,
    )
    user = UserService.create(db, user_in=user_in)
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "password": "adminpass123",
    }

@pytest.fixture(scope="function")
def user_token_headers(client, normal_user) -> Dict[str, str]:
    """
    Get token headers for a normal user.
    """
    login_data = {
        "username": normal_user["username"],
        "password": normal_user["password"],
    }
    response = client.post("/api/auth/login", data=login_data)
    tokens = response.json()
    access_token = tokens["access_token"]
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture(scope="function")
def superuser_token_headers(client, superuser) -> Dict[str, str]:
    """
    Get token headers for a superuser.
    """
    login_data = {
        "username": superuser["username"],
        "password": superuser["password"],
    }
    response = client.post("/api/auth/login", data=login_data)
    tokens = response.json()
    access_token = tokens["access_token"]
    return {"Authorization": f"Bearer {access_token}"} 