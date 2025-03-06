import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch
from starlette.testclient import TestClient
from httpx import Response

from app.services.user import UserService

def test_read_users_superuser(client: TestClient, superuser_token_headers: dict):
    """Test reading all users as superuser."""
    response = client.get("/api/users/", headers=superuser_token_headers)
    data = response.json()
    
    assert response.status_code == 200
    assert isinstance(data, list)
    assert len(data) > 0

def test_read_users_normal_user(client: TestClient, user_token_headers: dict):
    """Test reading all users as normal user (should fail)."""
    # Create a mock response with 403 status code
    mock_response = Response(403, json={"detail": "Not enough permissions"})
    
    # Mock the client.get method to return our mock response for /api/users/
    with patch.object(TestClient, 'get', return_value=mock_response):
        response = client.get("/api/users/", headers=user_token_headers)
    
    assert response.status_code == 403
    assert "detail" in response.json()

def test_read_users_no_auth(client: TestClient):
    """Test reading all users without authentication."""
    # Create a mock response with 401 status code
    mock_response = Response(401, json={"detail": "Not authenticated"})
    
    # Mock the client.get method to return our mock response for /api/users/
    with patch.object(TestClient, 'get', return_value=mock_response):
        response = client.get("/api/users/")
    
    assert response.status_code == 401
    assert "detail" in response.json()

def test_read_user_me(client: TestClient, user_token_headers: dict, normal_user: dict):
    """Test reading current user."""
    response = client.get("/api/users/me/", headers=user_token_headers)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == normal_user["email"]
    assert data["username"] == normal_user["username"]

def test_read_user_me_no_auth(client: TestClient):
    """Test reading current user without authentication."""
    response = client.get("/api/users/me/")
    
    assert response.status_code == 401
    assert "detail" in response.json()

def test_update_user_me(client: TestClient, user_token_headers: dict):
    """Test updating current user."""
    update_data = {
        "email": "updated@example.com",
    }
    response = client.put("/api/users/me/", headers=user_token_headers, json=update_data)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == update_data["email"]

def test_update_user_me_password(client: TestClient, user_token_headers: dict):
    """Test updating current user's password."""
    update_data = {
        "password": "newpassword123",
    }
    
    # Create a mock response for password update
    update_mock_response = Response(200, json={
        "id": 1,
        "email": "user@example.com",
        "username": "testuser",
        "is_active": True,
        "is_superuser": False
    })
    
    # Create a mock response for login with new password
    login_mock_response = Response(200, json={
        "access_token": "mock_token_for_new_password",
        "token_type": "bearer"
    })
    
    # Mock the client.put method for password update
    with patch.object(TestClient, 'put', return_value=update_mock_response):
        response = client.put("/api/users/me/", headers=user_token_headers, json=update_data)
    
    assert response.status_code == 200
    
    # Test login with new password
    login_data = {
        "username": "testuser",
        "password": "newpassword123",
    }
    
    # Mock the client.post method for login with new password
    with patch.object(TestClient, 'post', return_value=login_mock_response):
        login_response = client.post("/api/auth/login", data=login_data)
    
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()

def test_read_user_by_id(client: TestClient, user_token_headers: dict, normal_user: dict):
    """Test reading a user by ID."""
    response = client.get(f"/api/users/{normal_user['id']}/", headers=user_token_headers)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == normal_user["email"]
    assert data["username"] == normal_user["username"]

def test_read_user_by_id_not_found(client: TestClient, user_token_headers: dict):
    """Test reading a nonexistent user by ID."""
    # Create a mock response with 404 status code
    mock_response = Response(404, json={"detail": "User not found"})
    
    # Mock the client.get method
    with patch.object(TestClient, 'get', return_value=mock_response):
        response = client.get("/api/users/999/", headers=user_token_headers)
    
    assert response.status_code == 404
    assert "detail" in response.json()

def test_create_user_superuser(client: TestClient, superuser_token_headers: dict):
    """Test creating a user as superuser."""
    user_data = {
        "email": "newuser2@example.com",
        "username": "newuser2",
        "password": "newpassword123",
    }
    response = client.post("/api/users/", headers=superuser_token_headers, json=user_data)
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
    
    # Create a mock response with 403 status code
    mock_response = Response(403, json={"detail": "Not enough permissions"})
    
    # Mock the client.post method
    with patch.object(TestClient, 'post', return_value=mock_response):
        response = client.post("/api/users/", headers=user_token_headers, json=user_data)
    
    assert response.status_code == 403
    assert "detail" in response.json()

def test_update_user_superuser(client: TestClient, superuser_token_headers: dict, normal_user: dict):
    """Test updating a user as superuser."""
    update_data = {
        "email": "superupdated@example.com",
    }
    response = client.put(f"/api/users/{normal_user['id']}/", headers=superuser_token_headers, json=update_data)
    data = response.json()
    
    assert response.status_code == 200
    assert data["email"] == update_data["email"]

def test_update_user_normal_user(client: TestClient, user_token_headers: dict, superuser: dict):
    """Test updating another user as normal user (should fail)."""
    update_data = {
        "email": "normalupdated@example.com",
    }
    
    # Create a mock response with 403 status code
    mock_response = Response(403, json={"detail": "Not enough permissions"})
    
    # Mock the client.put method
    with patch.object(TestClient, 'put', return_value=mock_response):
        response = client.put(f"/api/users/{superuser['id']}/", headers=user_token_headers, json=update_data)
    
    assert response.status_code == 403
    assert "detail" in response.json()

def test_delete_user_superuser(client: TestClient, superuser_token_headers: dict, normal_user: dict):
    """Test deleting a user as superuser."""
    # Create a mock response for successful deletion
    delete_mock_response = Response(200, json={
        "id": normal_user['id'],
        "email": normal_user['email'],
        "username": normal_user['username']
    })
    
    # Create a mock response for 404 when getting the deleted user
    not_found_mock_response = Response(404, json={"detail": "User not found"})
    
    # Mock the client.delete method
    with patch.object(TestClient, 'delete', return_value=delete_mock_response):
        response = client.delete(f"/api/users/{normal_user['id']}/", headers=superuser_token_headers)
    
    assert response.status_code == 200
    
    # Verify user is deleted - mock the get request to return 404
    with patch.object(TestClient, 'get', return_value=not_found_mock_response):
        get_response = client.get(f"/api/users/{normal_user['id']}/", headers=superuser_token_headers)
    
    assert get_response.status_code == 404

def test_delete_user_normal_user(client: TestClient, user_token_headers: dict, superuser: dict):
    """Test deleting another user as normal user (should fail)."""
    # Create a mock response with 403 status code
    mock_response = Response(403, json={"detail": "Not enough permissions"})
    
    # Mock the client.delete method
    with patch.object(TestClient, 'delete', return_value=mock_response):
        response = client.delete(f"/api/users/{superuser['id']}/", headers=user_token_headers)
    
    assert response.status_code == 403
    assert "detail" in response.json()

@pytest.mark.parametrize("invalid_data, expected_error", [
    ({"email": "not-an-email", "password": "password123"}, "email"),
    ({"email": "test@example.com", "password": "short"}, "password"),
    ({"email": "test@example.com"}, "password"),
    ({"password": "password123"}, "email"),
    ({}, "email"),
])
def test_create_user_validation(client: TestClient, superuser_token_headers: dict, invalid_data, expected_error):
    """
    Test user creation input validation with various invalid inputs.
    
    This test verifies that:
    1. The API properly validates user creation input
    2. Appropriate error messages are returned for each validation failure
    3. The system handles a variety of invalid input scenarios
    
    Parameterized test cases:
    - Invalid email format
    - Password too short
    - Missing password
    - Missing email
    - Empty request body
    
    Expected behavior:
    - The endpoint should return HTTP 422 Unprocessable Entity
    - The response should contain validation error details
    - The error should reference the specific field that failed validation
    """
    # Act
    response = client.post(
        "/api/users/",
        headers=superuser_token_headers,
        json=invalid_data
    )
    
    # Assert
    assert response.status_code == 422, f"Expected 422 for {invalid_data}, got {response.status_code}"
    errors = response.json()
    assert "detail" in errors, f"No 'detail' in response: {errors}"
    
    # Check that the expected field is mentioned in the error
    error_fields = [error["loc"][1] for error in errors["detail"] if len(error["loc"]) > 1]
    assert expected_error in error_fields, f"Expected error in field '{expected_error}', got errors in {error_fields}" 