# JKT Application - Quick Start Guide

This quick start guide will help you get the JKT application up and running in minutes.

## Prerequisites

- Python 3.10+
- Git
- SQLite (included) or PostgreSQL

## Setup in 5 Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd JKT_EX
```

2. **Set up Python environment**

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure environment**

Create a `.env` file in the project root:

```
# Minimal configuration (SQLite)
SECRET_KEY=your_secret_key_here
CSRF_KEY=your_csrf_key_here
DATABASE_URL=sqlite:///./app.db
```

4. **Initialize the database**

```bash
# Run migrations
alembic upgrade head
```

5. **Start the application**

```bash
uvicorn app.main:app --reload
```

The application will be available at http://localhost:8000

## Quick Test

Verify the setup by running a basic test:

```bash
pytest tests/test_api/test_users.py -v
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Common Operations

### Create a User

```bash
curl -X POST "http://localhost:8000/api/users/" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"securepassword","username":"testuser"}'
```

### Login

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"securepassword"}'
```

### View Your Profile

```bash
curl -X GET "http://localhost:8000/api/users/me" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Next Steps

- Read the full [README.md](README.md) for more details
- Check [tests/README_TESTING.md](tests/README_TESTING.md) for testing guidelines
- Explore the [Project Structure](#project-structure)

## Project Structure

```
JKT_EX/
├── app/                  # Main application code
│   ├── api/              # API routes
│   ├── core/             # Core configuration
│   ├── db/               # Database
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   └── main.py           # Application entry point
├── tests/                # Test suite
└── alembic/              # Database migrations
```

## Troubleshooting

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