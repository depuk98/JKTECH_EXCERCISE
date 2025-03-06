# Testing Guide for JKT Application

This guide provides detailed information about the testing infrastructure, methodologies, and best practices for the JKT application.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Test Directory Structure](#test-directory-structure)
- [Test Types](#test-types)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [API Tests](#api-tests)
  - [Database Tests](#database-tests)
  - [Performance Tests](#performance-tests)
- [Test Fixtures](#test-fixtures)
- [Running Tests](#running-tests)
- [Writing New Tests](#writing-new-tests)
- [Testing Async Code](#testing-async-code)
- [Common Testing Patterns](#common-testing-patterns)
- [Troubleshooting](#troubleshooting)

## Testing Overview

The JKT application uses pytest as its testing framework with additional plugins for asynchronous testing and coverage reporting. The test suite is designed to provide comprehensive coverage across all application layers.

## Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py                # Shared test fixtures
├── README_TESTING.md          # This file
├── test_performance.py        # Legacy performance tests
├── test_api/                  # API endpoint tests
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_documents.py
│   └── test_users.py
├── test_db/                   # Database model tests
│   ├── __init__.py
│   └── test_crud.py
├── test_integration/          # Integration tests
│   ├── __init__.py
│   └── test_workflows.py
├── test_performance/          # Performance and concurrency tests
│   ├── __init__.py
│   └── test_concurrency.py
└── test_services/             # Service layer tests
    ├── __init__.py
    ├── test_document_service.py
    └── test_user_service.py
```

## Test Types

### Unit Tests

Unit tests focus on testing individual components in isolation, typically mocking external dependencies.

**Example:**
```python
def test_password_hashing():
    password = "testpassword"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed)
```

### Integration Tests

Integration tests verify that different components work together correctly.

**Example:**
```python
def test_user_document_integration(db):
    # Test that user and document relationships work correctly
    user = create_test_user(db)
    document = create_test_document(db, user_id=user.id)
    assert document in user.documents
```

### API Tests

API tests verify the behavior of API endpoints using FastAPI's TestClient.

**Example:**
```python
def test_create_user(client):
    response = client.post(
        "/api/users/",
        json={"email": "test@example.com", "password": "password", "username": "testuser"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
```

### Database Tests

Database tests focus on model interactions and database queries.

**Example:**
```python
def test_create_user_model(db):
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashedpassword",
        is_active=True
    )
    db.add(user)
    db.commit()
    
    db_user = db.query(User).filter(User.email == "test@example.com").first()
    assert db_user is not None
    assert db_user.username == "testuser"
```

### Performance Tests

Performance tests evaluate the application's ability to handle load and concurrent operations.

**Example:**
```python
@pytest.mark.asyncio
async def test_concurrent_write_operations(db):
    # Test concurrent database writes
    # Create tasks for multiple simultaneous operations
    tasks = [upload_document(user_id, token, f"test_doc_{j}") for j in range(10)]
    document_ids = await asyncio.gather(*tasks)
    
    # Verify all operations succeeded
    assert len(document_ids) == 10
    assert all(doc_id is not None for doc_id in document_ids)
```

## Test Fixtures

The `conftest.py` file contains shared fixtures used across tests. Key fixtures include:

1. **db**: Provides a SQLAlchemy session for database operations
2. **async_db**: Provides an async SQLAlchemy session
3. **client**: Provides a FastAPI TestClient for API testing
4. **normal_user**: Creates a regular test user
5. **superuser**: Creates a test superuser
6. **user_token_headers**: Provides authentication headers for a normal user
7. **superuser_token_headers**: Provides authentication headers for a superuser

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app

# Run tests in a specific directory
pytest tests/test_api/

# Run a specific test file
pytest tests/test_api/test_users.py

# Run a specific test function
pytest tests/test_api/test_users.py::test_create_user

# Run tests matching a pattern
pytest -k "create or update"
```

### Performance Testing

```bash
# Run all performance tests
pytest tests/test_performance/

# Run concurrency tests
pytest tests/test_performance/test_concurrency.py

# Run a specific concurrency test
pytest tests/test_performance/test_concurrency.py::test_concurrent_write_operations
```

## Writing New Tests

When writing new tests, follow these guidelines:

1. **Place tests in the appropriate directory** based on what they're testing
2. **Use the existing fixtures** from conftest.py when possible
3. **Follow the naming convention** `test_*.py` for files and `test_*` for functions
4. **Clean up after your tests** to ensure test isolation
5. **For async tests**, use the `@pytest.mark.asyncio` decorator

### Example Test Structure

```python
import pytest
from app.models.user import User

def test_create_user(db):
    """Test creating a user in the database."""
    # Arrange
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "hashed_password": "hashedpassword",
        "is_active": True
    }
    
    # Act
    user = User(**user_data)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Assert
    assert user.id is not None
    assert user.email == user_data["email"]
    assert user.username == user_data["username"]
    
    # Clean up (handled by db fixture)
```

## Testing Async Code

For testing asynchronous code, use the `pytest-asyncio` plugin. Mark your test functions with `@pytest.mark.asyncio` and use the `async_db` fixture:

```python
@pytest.mark.asyncio
async def test_async_operation(async_db):
    # Arrange
    # ...

    # Act - perform some async operation
    result = await my_async_function()

    # Assert
    assert result is not None
```

## Common Testing Patterns

### Testing Database Operations

```python
def test_database_operation(db):
    # Create test data
    # Perform operation
    # Verify results
```

### Testing API Endpoints

```python
def test_api_endpoint(client, user_token_headers):
    # Make request
    response = client.get("/api/endpoint", headers=user_token_headers)
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["key"] == "expected_value"
```

### Testing Concurrent Operations

```python
@pytest.mark.asyncio
async def test_concurrent_operations(db):
    # Create multiple tasks
    tasks = [async_operation() for _ in range(10)]
    # Run concurrently
    results = await asyncio.gather(*tasks)
    # Verify results
    assert len(results) == 10
```

## Troubleshooting

### No Such Table Error

If you encounter a "no such table" error:

1. Check if your test is using the right database session (e.g., `db` fixture)
2. Verify that the database tables are created before your test runs
3. Make sure your model is correctly defined and imported

### SQLite Concurrency Issues

SQLite has limitations with concurrent write operations:

1. Use separate database sessions for each concurrent task
2. Use proper transaction isolation
3. Consider using PostgreSQL for testing complex concurrent operations

### Async Test Failures

If async tests are failing:

1. Ensure you're using the `@pytest.mark.asyncio` decorator
2. Use the correct async fixtures (`async_db` instead of `db`)
3. Make sure async functions are properly awaited
4. Check that async database operations use the correct SQLAlchemy 2.0 syntax
5. Verify that async sessions are properly closed after use 