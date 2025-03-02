# Test Suite Troubleshooting Guide

This guide helps you run and troubleshoot the test suite.

## Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-asyncio
```

## Test Structure

- `tests/test_api/`: Tests for API endpoints
- `tests/test_services/`: Tests for service layer components
- `tests/test_document_service.py`: Legacy tests for document service
- `tests/conftest.py`: Contains pytest fixtures and configuration

## Common Test Issues

### 1. Missing Fixtures

Ensure you have all fixtures properly configured in `conftest.py`. The tests rely on fixtures for:
- Database sessions
- Authentication tokens
- Mock documents and users

### 2. Mock Objects Configuration

When mocking SQLAlchemy objects:
- For `db.query` tests, mock the query chain completely
- For `db.execute` tests, mock the return value appropriate for the query type
- Use `reset_mock()` to clear previous calls when reusing mocks

### 3. Asynchronous Testing

- Use `@pytest.mark.asyncio` decorator for async tests
- Use `AsyncMock` for mocking async functions

### 4. Common Test Errors

#### Missing `pytest.ANY`

If you see errors about `pytest.ANY`, import it from unittest.mock:
```python
from unittest.mock import ANY
```

#### Mock Object Attribute Access

Errors like "Mock object has no attribute 'X'" are usually due to incomplete mock setup.
Ensure you define all required attributes:

```python
# Example proper setup for an UploadFile mock
upload_file = MagicMock(spec=UploadFile)
upload_file.file = MagicMock()  # Create file attribute
upload_file.file.read = MagicMock(return_value=content)
```

#### Database Query Mocking

Properly mock database queries:

```python
# For query style:
mock_db.query = MagicMock()
mock_db.query.return_value.filter.return_value.first.return_value = mock_document

# For execute style:
mock_db.execute = MagicMock()
mock_db.execute.return_value = mock_results
```

## Running Tests

Run specific test files:

```bash
# Run API document tests
pytest tests/test_api/test_documents.py -v

# Run service tests
pytest tests/test_services/test_document_deletion.py -v
```

Run with coverage:

```bash
# Run all tests with coverage
pytest --cov=app

# Generate HTML coverage report
pytest --cov=app --cov-report=html
```

## Test Code Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Use appropriate mocking to isolate the component under test
3. **Assertions**: Make specific assertions about expected outcomes 
4. **Coverage**: Use coverage reports to identify untested code
5. **Descriptive Names**: Use descriptive test names that explain what is being tested 