import time
import pytest
import statistics
from fastapi.testclient import TestClient

def measure_response_time(client, url, method="get", headers=None, json=None, data=None):
    """Measure response time for a request."""
    start_time = time.time()
    
    if method.lower() == "get":
        response = client.get(url, headers=headers)
    elif method.lower() == "post":
        response = client.post(url, headers=headers, json=json, data=data)
    elif method.lower() == "put":
        response = client.put(url, headers=headers, json=json)
    elif method.lower() == "delete":
        response = client.delete(url, headers=headers)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return response, response_time

def test_login_performance(client: TestClient, normal_user: dict):
    """Test login endpoint performance."""
    login_data = {
        "username": normal_user["username"],
        "password": normal_user["password"],
    }
    
    # Perform multiple requests to get average response time
    response_times = []
    for _ in range(10):
        _, response_time = measure_response_time(
            client, "/api/auth/login", method="post", data=login_data
        )
        response_times.append(response_time)
    
    avg_response_time = statistics.mean(response_times)
    max_response_time = max(response_times)
    min_response_time = min(response_times)
    
    print(f"\nLogin Performance:")
    print(f"  Average: {avg_response_time:.4f} seconds")
    print(f"  Maximum: {max_response_time:.4f} seconds")
    print(f"  Minimum: {min_response_time:.4f} seconds")
    
    # Assert that the average response time is below a threshold
    assert avg_response_time < 0.5  # 500ms threshold

def test_register_performance(client: TestClient):
    """Test register endpoint performance."""
    # Perform multiple requests to get average response time
    response_times = []
    for i in range(10):
        user_data = {
            "email": f"perftest{i}@example.com",
            "username": f"perftest{i}",
            "password": "password123",
        }
        
        _, response_time = measure_response_time(
            client, "/api/auth/register", method="post", json=user_data
        )
        response_times.append(response_time)
    
    avg_response_time = statistics.mean(response_times)
    max_response_time = max(response_times)
    min_response_time = min(response_times)
    
    print(f"\nRegister Performance:")
    print(f"  Average: {avg_response_time:.4f} seconds")
    print(f"  Maximum: {max_response_time:.4f} seconds")
    print(f"  Minimum: {min_response_time:.4f} seconds")
    
    # Assert that the average response time is below a threshold
    assert avg_response_time < 0.5  # 500ms threshold

def test_get_user_me_performance(client: TestClient, user_token_headers: dict):
    """Test get current user endpoint performance."""
    # Perform multiple requests to get average response time
    response_times = []
    for _ in range(10):
        _, response_time = measure_response_time(
            client, "/api/users/me", headers=user_token_headers
        )
        response_times.append(response_time)
    
    avg_response_time = statistics.mean(response_times)
    max_response_time = max(response_times)
    min_response_time = min(response_times)
    
    print(f"\nGet User Me Performance:")
    print(f"  Average: {avg_response_time:.4f} seconds")
    print(f"  Maximum: {max_response_time:.4f} seconds")
    print(f"  Minimum: {min_response_time:.4f} seconds")
    
    # Assert that the average response time is below a threshold
    assert avg_response_time < 0.5  # 500ms threshold

def test_get_users_performance(client: TestClient, superuser_token_headers: dict):
    """Test get all users endpoint performance."""
    # Perform multiple requests to get average response time
    response_times = []
    for _ in range(10):
        _, response_time = measure_response_time(
            client, "/api/users", headers=superuser_token_headers
        )
        response_times.append(response_time)
    
    avg_response_time = statistics.mean(response_times)
    max_response_time = max(response_times)
    min_response_time = min(response_times)
    
    print(f"\nGet All Users Performance:")
    print(f"  Average: {avg_response_time:.4f} seconds")
    print(f"  Maximum: {max_response_time:.4f} seconds")
    print(f"  Minimum: {min_response_time:.4f} seconds")
    
    # Assert that the average response time is below a threshold
    assert avg_response_time < 0.5  # 500ms threshold 