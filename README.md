# 🚀 JKT Application

A modern web application built with FastAPI, SQLAlchemy, and advanced testing.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## 🔍 Overview

JKT is a robust backend service with an API for user management, document processing, and data retrieval. It features a comprehensive test suite to ensure reliability and performance.

## ✨ Features

- 🔐 User authentication and management
- 📄 Document upload and processing
- 🌐 RESTful API with OpenAPI documentation
- ⚡ Asynchronous operations for improved performance
- 🧪 Comprehensive test coverage
- 🔄 Concurrent operation handling
- 🔍 Vector embeddings for semantic search
- 📚 Hierarchical text chunking with sliding window 
- 🎨 Modern UI with Bootstrap

![Application Screenshot](docs/images/app_screenshot.png)

## 🚀 Quick Start

Get up and running in 5 minutes:

```bash
# Clone the repository
git clone <repository-url>
cd JKT_EX

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



# Initialize database
alembic upgrade head

# Start the application
uvicorn app.main:app --reload
```

The application will be available at http://localhost:8000

## 📦 Installation

### Prerequisites

- Python 3.10+ (recommended 3.11+)
- PostgreSQL (for production) or SQLite (for development)
- Git

### Detailed Installation Steps

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


## 🚀 Running the Application

1. **Run database migrations**

```bash
alembic upgrade head
```

2. **Start the development server**

```bash
uvicorn app.main:app --reload
```

The application will be available at http://localhost:8000

## 📚 API Documentation

- 🔍 Swagger UI: http://localhost:8000/docs
- 📖 ReDoc: http://localhost:8000/redoc

### Common API Operations

#### Create a User

```bash
curl -X POST "http://localhost:8000/api/users/" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"securepassword","username":"testuser"}'
```

#### Login

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"securepassword"}'
```

#### View Your Profile

```bash
curl -X GET "http://localhost:8000/api/users/me" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 📁 Project Structure

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
├── DEVELOPMENT.md        # Development guide
└── TESTING.md            # Testing guide
```

## 📝 Documentation

- 📖 [DEVELOPMENT.md](DEVELOPMENT.md) - Detailed architecture and development guide
- 🧪 [TESTING.md](TESTING.md) - Comprehensive testing guide

## ❓ Troubleshooting

### Database Errors

If you see database errors, try:

```bash
# Reset the database
rm app.db
alembic upgrade head
```

### Import Errors

If you encounter import errors:

```bash
# Verify your Python path includes the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Port Already in Use

If port 8000 is already in use:

```bash
# Specify a different port
uvicorn app.main:app --reload --port 8080
``` 