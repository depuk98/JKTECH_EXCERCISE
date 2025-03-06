# JKT Application - Testing Guide

This guide provides comprehensive information about testing the JKT application, including test types, setup, execution, and troubleshooting.

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Test Directory Structure](#test-directory-structure)
3. [Test Types](#test-types)
   - [Unit Tests](#unit-tests)
   - [Integration Tests](#integration-tests)
   - [API Tests](#api-tests)
   - [Database Tests](#database-tests)
   - [Performance Tests](#performance-tests)
4. [Test Fixtures](#test-fixtures)
5. [Setting Up Local Testing Environment](#setting-up-local-testing-environment)
6. [Running Tests](#running-tests)
7. [Writing New Tests](#writing-new-tests)
8. [Testing Async Code](#testing-async-code)
9. [Automated Testing](#automated-testing)
   - [Git Hooks](#git-hooks)
10. [Common Testing Patterns](#common-testing-patterns)
11. [Troubleshooting](#troubleshooting)

## Testing Overview

The JKT application uses pytest as its testing framework with additional plugins for asynchronous testing and coverage reporting. The test suite is designed to provide comprehensive coverage across all application layers.

## Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py                # Shared test fixtures
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

### Database Fixtures

```python
@pytest.fixture(scope="function")
def db() -> Generator:
    """
    Create a fresh database for each test function.
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

@pytest_asyncio.fixture(scope="function")
async def async_db() -> AsyncGenerator:
    """
    Create a fresh async database for each test function.
    """
    # Create the database tables synchronously first
    Base.metadata.create_all(bind=engine)
    
    # Create a new async session for testing
    async with AsyncTestingSessionLocal() as session:
        yield session
    
    # Drop the database tables
    Base.metadata.drop_all(bind=engine)
```

### Client Fixture

```python
@pytest.fixture
def client(db):
    """
    Create a test client with DB overrides.
    """
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

### Authentication Fixtures

```python
@pytest.fixture
def normal_user(db) -> User:
    """Create a normal test user."""
    user = create_test_user(db)
    return user

@pytest.fixture
def user_token_headers(client, normal_user) -> Dict[str, str]:
    """Get token headers for a normal user."""
    return get_auth_headers_for_user(client, normal_user.email, "password")

@pytest.fixture
def superuser_token_headers(client, superuser) -> Dict[str, str]:
    """Get token headers for a superuser."""
    return get_auth_headers_for_user(client, superuser.email, "password")
```

## Setting Up Local Testing Environment

To run tests locally on your development machine, follow these steps:

### 1. Install Testing Dependencies

Make sure you have all required testing packages installed:

```bash
# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install testing dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio
```



### 2. Set Up Test Database

The tests will automatically set up and tear down the test database (SQLite), but you can manually prepare it if needed:

```bash
# Make sure no old test database exists
rm -f test.db
```

### 3. Clear Cached Test Results (Optional)

If you've run tests before, you might want to clear cached results:

```bash
# Remove pytest cache
rm -rf .pytest_cache
```

### 4. Complete Example: Running Tests Locally

Here's a complete example of setting up and running tests from scratch:

```bash
# 1. Navigate to the project directory
cd /path/to/JKT_EX

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio

# 4. Configure test environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
export APP_ENV=test

# 5. Run a specific test
python -m pytest tests/test_api/test_users.py -v

# 6. Run all tests with coverage report
python -m pytest --cov=app --cov-report=term
```

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

### Proper Async Session Management

When working with async SQLAlchemy sessions in tests, always use one of these patterns:

#### 1. Async Context Manager (Recommended)

```python
@pytest.mark.asyncio
async def test_async_db_operations():
    async with AsyncTestingSessionLocal() as session:
        # Do work with session
        # Session is automatically closed correctly
```

#### 2. Try/Finally Pattern

```python
@pytest.mark.asyncio
async def test_async_db_operations():
    session = AsyncTestingSessionLocal()
    try:
        # Do work with session
        return result
    finally:
        await session.close()  # Must await the close!
```

### Fixing "Coroutine was never awaited" Warnings

If you see this warning:

```
RuntimeWarning: coroutine 'AsyncSession.close' was never awaited
```

Make sure you're properly awaiting all async methods:

```python
# WRONG - will cause warning
session = AsyncTestingSessionLocal()
session.close()  # This is a coroutine that should be awaited

# CORRECT
session = AsyncTestingSessionLocal()
await session.close()
```

## Automated Testing

### Git Hooks

Git hooks are scripts that run automatically when certain Git events occur, like committing or pushing.

#### Setting Up Git Hooks

1. Create pre-commit and post-commit hook scripts:

```bash
# pre-commit-hook.sh
#!/bin/bash
set -e

echo "Running tests before commit..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests
python -m pytest tests/test_api/ -v

# If tests pass, allow the commit
echo "Tests passed! Proceeding with commit..."
exit 0
```

```bash
# post-commit-hook.sh
#!/bin/bash
set -e

echo "Running after commit actions..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run additional tasks after commit
# For example, generate coverage report
python -m pytest --cov=app --cov-report=term

echo "Post-commit tasks completed!"
exit 0
```

2. Install the hooks:

```bash
# Make the hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy the pre-commit hook
cp pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Copy the post-commit hook
cp post-commit-hook.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit
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

### Mocking External Services

```python
@patch("app.services.external_service.make_api_call")
def test_with_mock(mock_api_call, db):
    # Configure mock
    mock_api_call.return_value = {"status": "success"}
    
    # Call function that uses the external service
    result = my_function_that_calls_external_service()
    
    # Verify mock was called correctly
    mock_api_call.assert_called_once_with(expected_args)
    
    # Verify result
    assert result == expected_result
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

Example fix for concurrent operations with SQLite:

```python
@pytest.mark.asyncio
async def test_concurrent_write_operations(db):
    # Create test users
    user_ids = []
    # ... create test users ...
    
    # Helper function with proper session management
    async def upload_document(user_id):
        # Create a new session for each task
        task_db = TestingSessionLocal()
        try:
            document = Document(
                user_id=user_id,
                filename=f"test_{user_id}.pdf",
                content_type="application/pdf",
                status="uploaded"
            )
            task_db.add(document)
            task_db.commit()
            task_db.refresh(document)
            document_id = document.id
            return document_id
        finally:
            task_db.close()
    
    # Create tasks
    tasks = []
    for user_id in user_ids:
        for j in range(3):
            tasks.append(upload_document(user_id))
    
    # Run tasks concurrently
    document_ids = await asyncio.gather(*tasks)
    
    # Verify results
    assert len(document_ids) == len(user_ids) * 3
```

### Async Test Failures

If async tests are failing:

1. Ensure you're using the `@pytest.mark.asyncio` decorator
2. Use the correct async fixtures (`async_db` instead of `db`)
3. Make sure async functions are properly awaited
4. Check that async database operations use the correct SQLAlchemy 2.0 syntax
5. Verify that async sessions are properly closed after use

### Debugging Tips

1. **Enable warnings as errors** during development:
   ```bash
   python -W error::RuntimeWarning test_your_file.py
   ```

2. **Add stack trace to warnings** to see exactly where they originate:
   ```python
   import warnings
   warnings.filterwarnings('always', category=RuntimeWarning, append=True)
   ```

3. **Use `asyncio.create_task` with caution** and always await the tasks:
   ```python
   task = asyncio.create_task(some_coroutine())
   await task  # Don't forget this!
   ```

4. **Review your test teardown** logic to ensure all async resources are properly closed. 