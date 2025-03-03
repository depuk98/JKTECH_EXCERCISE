import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.services.user import UserService

def test_read_users_superuser(client: TestClient, superuser_token_headers: dict):
    """Test reading all users as superuser."""
    response = client.get("/api/users", headers=superuser_token_headers)
    data = response.json()
    
    assert response.status_code == 200
    assert isinstance(data, list)
    assert len(data) > 0

def test_read_users_normal_user(client: TestClient, user_token_headers: dict):
    """Test reading all users as normal user (should fail)."""
    response = client.get("/api/users", headers=user_token_headers)
    
    assert response.status_code == 403
    assert "detail" in response.json()

def test_read_users_no_auth(client: TestClient):
    """Test reading all users without authentication."""
    response = client.get("/api/users")
    
    assert response.status_code == 401
    assert "detail" in response.json()

def test_read_user_me(client: TestClient, user_token_headers: dict, normal_user: dict):
    """Test reading current user."""
    response = client.get("/api/users/me", headers=user_token_headers)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == normal_user["email"]
    assert data["username"] == normal_user["username"]

def test_read_user_me_no_auth(client: TestClient):
    """Test reading current user without authentication."""
    response = client.get("/api/users/me")
    
    assert response.status_code == 401
    assert "detail" in response.json()

def test_update_user_me(client: TestClient, user_token_headers: dict):
    """Test updating current user."""
    update_data = {
        "email": "updated@example.com",
    }
    response = client.put("/api/users/me", headers=user_token_headers, json=update_data)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == update_data["email"]

def test_update_user_me_password(client: TestClient, user_token_headers: dict):
    """Test updating current user's password."""
    update_data = {
        "password": "newpassword123",
    }
    response = client.put("/api/users/me", headers=user_token_headers, json=update_data)
    
    assert response.status_code == 200
    
    # Test login with new password
    login_data = {
        "username": "testuser",
        "password": "newpassword123",
    }
    login_response = client.post("/api/auth/login", data=login_data)
    
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()

def test_read_user_by_id(client: TestClient, user_token_headers: dict, normal_user: dict):
    """Test reading a user by ID."""
    response = client.get(f"/api/users/{normal_user['id']}", headers=user_token_headers)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == normal_user["email"]
    assert data["username"] == normal_user["username"]

def test_read_user_by_id_not_found(client: TestClient, user_token_headers: dict):
    """Test reading a nonexistent user by ID."""
    response = client.get("/api/users/999", headers=user_token_headers)
    
    assert response.status_code == 404
    assert "detail" in response.json()

def test_create_user_superuser(client: TestClient, superuser_token_headers: dict):
    """Test creating a user as superuser."""
    user_data = {
        "email": "newuser2@example.com",
        "username": "newuser2",
        "password": "newpassword123",
    }
    response = client.post("/api/users", headers=superuser_token_headers, json=user_data)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == user_data["email"]
    assert data["username"] == user_data["username"]

def test_create_user_normal_user(client: TestClient, user_token_headers: dict):
    """Test creating a user as normal user (should fail)."""
    user_data = {
        "email": "newuser3@example.com",
        "username": "newuser3",
        "password": "newpassword123",
    }
    response = client.post("/api/users", headers=user_token_headers, json=user_data)
    
    assert response.status_code == 403
    assert "detail" in response.json()

def test_update_user_superuser(client: TestClient, superuser_token_headers: dict, normal_user: dict):
    """Test updating a user as superuser."""
    update_data = {
        "email": "superupdated@example.com",
    }
    response = client.put(f"/api/users/{normal_user['id']}", headers=superuser_token_headers, json=update_data)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == update_data["email"]

def test_update_user_normal_user(client: TestClient, user_token_headers: dict, superuser: dict):
    """Test updating another user as normal user (should fail)."""
    update_data = {
        "email": "normalupdated@example.com",
    }
    response = client.put(f"/api/users/{superuser['id']}", headers=user_token_headers, json=update_data)
    
    assert response.status_code == 403
    assert "detail" in response.json()

def test_delete_user_superuser(client: TestClient, superuser_token_headers: dict, normal_user: dict):
    """Test deleting a user as superuser."""
    response = client.delete(f"/api/users/{normal_user['id']}", headers=superuser_token_headers)
    
    assert response.status_code == 200
    
    # Verify user is deleted
    get_response = client.get(f"/api/users/{normal_user['id']}", headers=superuser_token_headers)
    assert get_response.status_code == 404

def test_delete_user_normal_user(client: TestClient, user_token_headers: dict, superuser: dict):
    """Test deleting another user as normal user (should fail)."""
    response = client.delete(f"/api/users/{superuser['id']}", headers=user_token_headers)
    
    assert response.status_code == 403
    assert "detail" in response.json() 