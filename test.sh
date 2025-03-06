#!/bin/bash
# Simple test runner for the Document Management and RAG-based Q&A Application
# Usage: ./test.sh [options]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
COVERAGE=false
TEST_TYPE="all"
FORMAT="text"
PARALLEL=false
VERBOSE=false

# Help function
function show_help {
    echo -e "${BLUE}Document Management and RAG-based Q&A Application Test Runner${NC}"
    echo
    echo "Usage: ./test.sh [options]"
    echo
    echo "Options:"
    echo "  -h, --help         Show this help message"
    echo "  -u, --unit         Run only unit tests"
    echo "  -i, --integration  Run only integration tests"
    echo "  -a, --api          Run only API tests"
    echo "  -p, --performance  Run only performance tests"
    echo "  -c, --coverage     Generate coverage report"
    echo "  --html             Generate HTML coverage report"
    echo "  --xml              Generate XML coverage report"
    echo "  --parallel         Run tests in parallel (when applicable)"
    echo "  -v, --verbose      Increase verbosity"
    echo
    echo "Examples:"
    echo "  ./test.sh                    # Run all tests"
    echo "  ./test.sh -u -c              # Run unit tests with coverage"
    echo "  ./test.sh -i -p --html       # Run integration and performance tests with HTML coverage"
    echo "  ./test.sh --parallel -v      # Run all tests in parallel with verbose output"
    echo
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--unit)
            if [ "$TEST_TYPE" = "all" ]; then
                TEST_TYPE="unit"
            else
                TEST_TYPE="$TEST_TYPE unit"
            fi
            shift
            ;;
        -i|--integration)
            if [ "$TEST_TYPE" = "all" ]; then
                TEST_TYPE="integration"
            else
                TEST_TYPE="$TEST_TYPE integration"
            fi
            shift
            ;;
        -a|--api)
            if [ "$TEST_TYPE" = "all" ]; then
                TEST_TYPE="api"
            else
                TEST_TYPE="$TEST_TYPE api"
            fi
            shift
            ;;
        -p|--performance)
            if [ "$TEST_TYPE" = "all" ]; then
                TEST_TYPE="performance"
            else
                TEST_TYPE="$TEST_TYPE performance"
            fi
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --html)
            COVERAGE=true
            FORMAT="html"
            shift
            ;;
        --xml)
            COVERAGE=true
            FORMAT="xml"
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Build python command
PYTHON_CMD="python run_tests.py"

# Add test types
if [ "$TEST_TYPE" != "all" ]; then
    for type in $TEST_TYPE; do
        PYTHON_CMD="$PYTHON_CMD --$type"
    done
else
    PYTHON_CMD="$PYTHON_CMD --all"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --coverage"
fi

# Add format
if [ "$FORMAT" != "text" ]; then
    PYTHON_CMD="$PYTHON_CMD --format $FORMAT"
fi

# Add parallel
if [ "$PARALLEL" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --parallel"
fi

# Add verbose
if [ "$VERBOSE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --verbose"
fi

# Display command
echo -e "${YELLOW}Running: $PYTHON_CMD${NC}"
echo

# Execute command
$PYTHON_CMD

# Get exit code
EXIT_CODE=$?

# Final message
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Tests failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE 