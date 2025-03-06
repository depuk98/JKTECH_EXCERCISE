#!/usr/bin/env python
"""
Enhanced test runner for the Document Management and RAG-based Q&A Application.

This script provides various options for running tests:
- Run specific test categories
- Generate coverage reports
- Run tests in parallel
- Output test results in different formats

Usage:
    python run_tests.py [OPTIONS]

Options:
    --unit           Run only unit tests
    --integration    Run only integration tests
    --api            Run only API tests
    --performance    Run only performance tests
    --all            Run all tests (default)
    --coverage       Generate coverage report
    --parallel       Run tests in parallel
    --format FORMAT  Output format (text, xml, html)
    --verbose        Increase verbosity
"""

import os
import sys
import argparse
import subprocess
import time
import asyncio
import platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Determine if we're on Windows for command formatting
IS_WINDOWS = platform.system() == "Windows"

# Define test categories
TEST_CATEGORIES = {
    "unit": ["test_services", "test_db"],
    "integration": ["test_integration"],
    "api": ["test_api"],
    "performance": ["test_performance"],
}

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Run tests for the application")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--format", choices=["text", "xml", "html"], default="text", 
                       help="Output format (text, xml, html)")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    return parser

def setup_test_environment():
    """Set up the test environment with required variables."""
    os.environ["TESTING"] = "1"
    os.environ["DATABASE_URL"] = "sqlite:///test.db"
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["LLM_MODEL"] = "llama3.2"

def build_command(args, test_paths=None):
    """Build the pytest command based on arguments."""
    cmd = ["pytest"]
    
    # Set verbosity
    if args.verbose > 0:
        cmd.extend(["-" + "v" * args.verbose])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=term"])
        if args.format == "xml":
            cmd.append("--cov-report=xml")
        elif args.format == "html":
            cmd.append("--cov-report=html")
    
    # Set output format
    if args.format == "xml":
        cmd.append("--junitxml=test-results.xml")
    
    # Add test paths if specified
    if test_paths:
        cmd.extend(test_paths)
    
    return cmd

def run_tests_sequentially(cmd):
    """Run tests sequentially."""
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    return result.returncode, elapsed

def run_tests_in_parallel(categories, args):
    """Run test categories in parallel using ThreadPoolExecutor."""
    start_time = time.time()
    commands = []
    
    # Create commands for each category
    for category, paths in categories.items():
        test_paths = [f"tests/{path}" for path in paths]
        cmd = build_command(args, test_paths)
        commands.append((category, cmd))
    
    results = []
    print(f"Running {len(commands)} test categories in parallel")
    
    # Execute commands in parallel
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = []
        for category, cmd in commands:
            print(f"Submitting {category} tests: {' '.join(cmd)}")
            future = executor.submit(subprocess.run, cmd, capture_output=args.verbose == 0)
            futures.append((category, future))
        
        # Collect results
        for category, future in futures:
            try:
                result = future.result()
                success = result.returncode == 0
                results.append((category, success, result))
                status = "Passed" if success else "Failed"
                print(f"{category.upper()} tests: {status}")
                if not success and args.verbose > 0:
                    print(f"Output for {category}:")
                    print(result.stdout.decode())
                    print(result.stderr.decode())
            except Exception as e:
                print(f"Error running {category} tests: {e}")
                results.append((category, False, None))
    
    # Calculate overall status
    overall_success = all(success for _, success, _ in results)
    elapsed = time.time() - start_time
    
    return 0 if overall_success else 1, elapsed

def main():
    """Main entry point for test runner."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If no test category specified, run all
    if not (args.unit or args.integration or args.api or args.performance or args.all):
        args.all = True
    
    # Set up environment
    setup_test_environment()
    
    # Determine which tests to run
    selected_categories = {}
    if args.all:
        for category, paths in TEST_CATEGORIES.items():
            selected_categories[category] = paths
    else:
        if args.unit:
            selected_categories["unit"] = TEST_CATEGORIES["unit"]
        if args.integration:
            selected_categories["integration"] = TEST_CATEGORIES["integration"]
        if args.api:
            selected_categories["api"] = TEST_CATEGORIES["api"]
        if args.performance:
            selected_categories["performance"] = TEST_CATEGORIES["performance"]
    
    print(f"Running tests with Python {sys.version}")
    
    if args.parallel and len(selected_categories) > 1:
        # Run tests in parallel
        return_code, elapsed = run_tests_in_parallel(selected_categories, args)
    else:
        # Run tests sequentially
        test_paths = []
        for paths in selected_categories.values():
            test_paths.extend([f"tests/{path}" for path in paths])
        
        cmd = build_command(args, test_paths)
        return_code, elapsed = run_tests_sequentially(cmd)
    
    # Print summary
    print(f"\nTest execution completed in {elapsed:.2f} seconds")
    print(f"Status: {'PASSED' if return_code == 0 else 'FAILED'}")
    
    # Generate reports if needed
    if args.coverage and args.format == "html":
        print("\nCoverage report generated at htmlcov/index.html")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main()) 