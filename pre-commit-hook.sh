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