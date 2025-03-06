#!/bin/bash
set -e

echo "Setting up Git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Install pre-commit hook
echo "Installing pre-commit hook..."
cp pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install post-commit hook
echo "Installing post-commit hook..."
cp post-commit-hook.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit

# Verify installation
echo "Installed hooks:"
ls -la .git/hooks/

echo "Git hooks setup complete!" 