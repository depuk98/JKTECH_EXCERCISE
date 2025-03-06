# Git Hooks for Automated Testing

This guide explains how to set up Git hooks to automatically run tests when you commit to the repository.

## What are Git Hooks?

Git hooks are scripts that Git executes before or after events such as: commit, push, and receive. They allow you to automate tasks that should run when these Git events occur.

## Available Hooks

This repository includes the following hook scripts:

- `pre-commit-hook.sh`: Runs API tests before allowing a commit
- `post-commit-hook.sh`: Generates a coverage report after a commit

## Setting Up Git Hooks

### Manual Setup

1. Copy the hook scripts to your `.git/hooks` directory:

```bash
# Make the hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy the pre-commit hook
cp pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Copy the post-commit hook
cp post-commit-hook.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

2. Verify the hooks are installed:

```bash
ls -la .git/hooks/
```

### Automatic Setup

You can use the setup script to install all hooks at once:

```bash
# Make the setup script executable
chmod +x setup-git-hooks.sh

# Run the setup script
./setup-git-hooks.sh
```

## Customizing Hooks

You can customize the hook scripts to run different tests or perform different actions:

1. Edit the script in the `.git/hooks` directory directly, or
2. Modify the script templates in this repository and reinstall the hooks

## Disabling Hooks Temporarily

To bypass hooks for a single commit:

```bash
git commit --no-verify -m "Your commit message"
```

## Troubleshooting

If you encounter issues with the hooks:

1. Ensure the hook scripts are executable (`chmod +x`)
2. Check that your Python environment is properly set up
3. Verify that pytest and other requirements are installed
4. Look for error messages in the terminal output

## Global Git Hooks

To use these hooks for all repositories on your system, you can configure global Git hooks:

```bash
git config --global core.hooksPath /path/to/your/hooks
```

This will use the hooks for all Git repositories on your system. 