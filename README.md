# JKT Application

A modern web application built with FastAPI, SQLAlchemy, and advanced testing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
  - [Test Types](#test-types)
  - [Running Tests](#running-tests)
  - [Performance Testing](#performance-testing)
  - [Troubleshooting Tests](#troubleshooting-tests)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Overview

This application provides a robust backend service with an API for user management, document processing, and data retrieval. It features a comprehensive test suite to ensure reliability and performance.

## Features

- User authentication and management
- Document upload and processing
- RESTful API
- Asynchronous operations
- Comprehensive test coverage
- Concurrent operation handling

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10+ (recommended 3.11+)
- PostgreSQL (for production) or SQLite (for development)
- git

## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd JKT_EX
```

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

On macOS/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
# Application
APP_ENV=development
DEBUG=True
SECRET_KEY=your_secret_key_here
CSRF_KEY=your_csrf_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=sqlite:///./app.db

# For PostgreSQL (production)
# DATABASE_URL=postgresql://username:password@localhost:5432/jkt_db

# Security
FRONTEND_HOST=http://localhost:3000
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

## Running the Application

1. **Run database migrations (if using SQL database)**

```bash
alembic upgrade head
```

2. **Start the development server**

```bash
uvicorn app.main:app --reload
```

The application will be available at http://localhost:8000

API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

The project includes a comprehensive test suite covering unit, integration, API, and performance tests.

### Test Types

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **API Tests**: Test API endpoints
- **DB Tests**: Test database operations
- **Performance Tests**: Test application performance and concurrency

### Running Tests

1. **Run all tests**

```bash
pytest
```

2. **Run specific test modules**

```bash
# Run API tests
pytest tests/test_api/

# Run DB tests
pytest tests/test_db/

# Run integration tests
pytest tests/test_integration/

# Run service tests
pytest tests/test_services/

# Run performance tests
pytest tests/test_performance/
```

3. **Run specific test files**

```bash
# Run a specific test file
pytest tests/test_api/test_users.py
```

4. **Run a specific test**

```bash
# Run a specific test function
pytest tests/test_performance/test_concurrency.py::test_concurrent_write_operations
```

5. **Run tests with verbose output**

```bash
pytest -v
```

6. **Run tests with increased log level**

```bash
pytest --log-cli-level=INFO
```

### Performance Testing

Performance tests verify the application's ability to handle concurrent operations and maintain data integrity under load:

```bash
# Run all performance tests
pytest tests/test_performance/

# Run concurrency tests
pytest tests/test_performance/test_concurrency.py
```

### Troubleshooting Tests

If you encounter database-related issues in tests:

1. **Check database configuration**

   Make sure the test database is properly configured in `tests/conftest.py`.

2. **SQLite concurrency issues**

   SQLite has limitations with concurrent write operations. For concurrent tests, consider:
   
   - Using separate database sessions for each task
   - Using a PostgreSQL database for testing concurrent operations
   - Ensuring proper transaction isolation

3. **Asynchronous test issues**

   - Ensure you're using the correct fixtures (e.g., `async_db` for async tests)
   - Check that async database operations use the correct syntax
   - Verify that async sessions are properly closed after use

## Project Structure

```
JKT_EX/
├── alembic/              # Database migrations
├── app/                  # Main application
│   ├── api/              # API endpoints and dependencies
│   ├── core/             # Core settings and security
│   ├── db/               # Database session and models
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   ├── templates/        # HTML templates
│   ├── static/           # Static files
│   ├── utils/            # Utility functions
│   ├── main.py           # Application entry point
│   └── __init__.py
├── tests/                # Test suite
│   ├── test_api/         # API tests
│   ├── test_db/          # Database tests
│   ├── test_integration/ # Integration tests
│   ├── test_performance/ # Performance/concurrency tests
│   ├── test_services/    # Service tests
│   ├── conftest.py       # Test fixtures and configuration
│   └── __init__.py
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (create this)
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request 