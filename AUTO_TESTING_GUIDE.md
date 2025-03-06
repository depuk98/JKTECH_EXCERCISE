# Automated Testing with Git Hooks and GitHub Actions

This guide explains how to set up automated testing for the JKT application using both local Git hooks and cloud-based GitHub Actions.

## Comparison: Git Hooks vs GitHub Actions

| Feature | Git Hooks | GitHub Actions |
|---------|-----------|----------------|
| Execution | Local machine | GitHub cloud |
| Setup | Manual per developer | Once for all developers |
| Configuration | Shell scripts | YAML workflow files |
| Trigger | Pre/post Git events | GitHub events |
| Environment | Developer's machine | Configurable VMs |
| Integration | Local only | GitHub ecosystem |
| Persistence | Local repository | GitHub repository |

## 1. Local Testing with Git Hooks

Git hooks are scripts that run on your local machine when certain Git events occur.

### Available Hook Scripts

- `pre-commit-hook.sh`: Run before committing to ensure tests pass
- `post-commit-hook.sh`: Run after committing to generate reports

### Setting Up Git Hooks

#### Manual Setup

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

#### Using the Setup Script

```bash
# Make the setup script executable
chmod +x setup-git-hooks.sh

# Run the setup script
./setup-git-hooks.sh
```

### Customizing Git Hooks

Edit the hook scripts to run specific tests or perform other actions:

```bash
# Example: running only API tests in pre-commit
python -m pytest tests/test_api/ -v
```

### Bypassing Git Hooks

To skip hooks for a single commit:

```bash
git commit --no-verify -m "Your commit message"
```

## 2. Cloud Testing with GitHub Actions

GitHub Actions run in GitHub's cloud environment when changes are pushed to your repository.

### Setting Up GitHub Actions

1. Create the workflows directory:

```bash
mkdir -p .github/workflows
```

2. Create a workflow file named `run-tests.yml`:

```yaml
name: Run Tests

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: |
        python -m pytest
    
    - name: Generate coverage report
      run: |
        python -m pytest --cov=app --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

3. Commit and push the workflow file:

```bash
git add .github/workflows/run-tests.yml
git commit -m "Add GitHub Actions workflow"
git push
```

### Viewing GitHub Actions Results

1. Go to the "Actions" tab in your GitHub repository
2. View workflow runs for recent pushes
3. Examine test results and logs

### Customizing GitHub Actions

You can customize the workflow to run specific tests:

```yaml
- name: Run specific tests
  run: |
    python -m pytest tests/test_performance/test_concurrency.py
```

## Best Practices for Continuous Integration

1. **Fast Tests First**: Run quick tests before slow ones
2. **Focused Tests**: Run only relevant tests when possible
3. **Consistent Environments**: Use the same environment for local and CI testing
4. **Clear Feedback**: Ensure test output is easy to understand
5. **Automate Everything**: Don't rely on manual steps
6. **Notifications**: Get alerts when tests fail
7. **Version Control**: Keep test configurations in version control

## Combined Approach

For the best results, use both approaches:

1. **Git Hooks**: For immediate local feedback before committing
2. **GitHub Actions**: For thorough testing in a clean environment

This ensures:
- Problems are caught early (local hooks)
- All tests run in a consistent environment (GitHub Actions)
- The team has visibility into test results (GitHub Actions)
- No broken code is merged (pull request checks) 