import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.services.user import UserService

def test_login_success(client: TestClient, normal_user: dict):
    """Test successful login."""
    login_data = {
        "username": normal_user["username"],
        "password": normal_user["password"],
    }
    response = client.post("/api/auth/login", data=login_data)
    tokens = response.json()
    
    assert response.status_code == 200
    assert "access_token" in tokens
    assert tokens["token_type"] == "bearer"

def test_login_fail_incorrect_password(client: TestClient, normal_user: dict):
    """Test login with incorrect password."""
    login_data = {
        "username": normal_user["username"],
        "password": "wrongpassword",
    }
    response = client.post("/api/auth/login", data=login_data)
    
    assert response.status_code == 401
    assert "detail" in response.json()

def test_login_fail_nonexistent_user(client: TestClient):
    """Test login with nonexistent user."""
    login_data = {
        "username": "nonexistentuser",
        "password": "password123",
    }
    response = client.post("/api/auth/login", data=login_data)
    
    assert response.status_code == 401
    assert "detail" in response.json()

def test_register_success(client: TestClient):
    """Test successful user registration."""
    user_data = {
        "email": "newuser@example.com",
        "username": "newuser",
        "password": "newpassword123",
    }
    response = client.post("/api/auth/register", json=user_data)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == user_data["email"]
    assert data["username"] == user_data["username"]
    assert "id" in data
    assert "hashed_password" not in data

def test_register_fail_duplicate_email(client: TestClient, normal_user: dict):
    """Test registration with duplicate email."""
    user_data = {
        "email": normal_user["email"],
        "username": "uniqueusername",
        "password": "password123",
    }
    response = client.post("/api/auth/register", json=user_data)
    
    assert response.status_code == 400
    assert "detail" in response.json()

def test_register_fail_duplicate_username(client: TestClient, normal_user: dict):
    """Test registration with duplicate username."""
    user_data = {
        "email": "unique@example.com",
        "username": normal_user["username"],
        "password": "password123",
    }
    response = client.post("/api/auth/register", json=user_data)
    
    assert response.status_code == 400
    assert "detail" in response.json()

def test_register_fail_invalid_username(client: TestClient):
    """Test registration with invalid username."""
    user_data = {
        "email": "valid@example.com",
        "username": "invalid username",  # Contains space
        "password": "password123",
    }
    response = client.post("/api/auth/register", json=user_data)
    
    assert response.status_code == 422
    assert "detail" in response.json()

def test_register_fail_short_password(client: TestClient):
    """Test registration with short password."""
    user_data = {
        "email": "valid@example.com",
        "username": "validuser",
        "password": "short",  # Less than 8 characters
    }
    response = client.post("/api/auth/register", json=user_data)
    
    assert response.status_code == 422
    assert "detail" in response.json() 