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