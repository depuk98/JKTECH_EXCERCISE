#!/bin/bash

# FastAPI Document Processing System Startup Script

# Detect virtual environment and activate if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Virtual environment not found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
        echo "Installing dependencies..."
        pip install -r requirements.txt
    fi
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using default configuration."
    echo "Consider creating a .env file for proper configuration."
fi

# Check PostgreSQL connection
echo "Checking database connection..."
db_url=$(grep DATABASE_URL .env 2>/dev/null | cut -d '=' -f2)
if [ -z "$db_url" ]; then
    echo "DATABASE_URL not found in .env file. Using default."
    db_host="localhost"
else
    db_host=$(echo $db_url | sed -n 's/.*@\([^:]*\).*/\1/p')
    if [ -z "$db_host" ]; then
        db_host="localhost"
    fi
fi

pg_isready -h $db_host &>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: PostgreSQL does not appear to be running on $db_host"
    echo "Please ensure PostgreSQL is running before proceeding."
    echo "Starting application anyway..."
else
    echo "PostgreSQL is running on $db_host"
fi

# Apply any pending migrations
echo "Running database migrations..."
alembic upgrade head

# Choose port
PORT=${1:-8000}
echo "Starting application on port $PORT..."

# Start the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port $PORT $2

# Note: Pass --reload as second argument for development mode
# Example: ./start_app.sh 8000 --reload 