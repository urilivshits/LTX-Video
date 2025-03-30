# Python Code Linting Guidelines

This project uses [ruff](https://github.com/astral-sh/ruff) for Python code linting to maintain consistent code quality across the codebase.

## Common Issues

The linting process checks for several types of issues, including:

1. **Module imports not at top of file (E402)**
   - All imports should be placed at the beginning of the file
   - Example fix: Move `import os` statements to the top section

2. **Unused imports (F401)**
   - Imported modules that aren't used in the code should be removed
   - Example fix: Remove `import json` if json isn't used

3. **F-strings without variables (F541)**
   - Replace `f"Text without variables"` with regular strings: `"Text without variables"`
   - Only use f-strings when they contain variables like `f"Value: {variable}"`

## How to Fix Linting Issues

### Automatic Fixes

Most linting issues can be fixed automatically with the included script:

```bash
# First, install ruff if you haven't already
pip install ruff

# Then run the automatic fixing script
python USER_DIR/fix_linting.py
```

This script will:
1. Check all Python files in USER_DIR for linting issues
2. Apply automatic fixes wherever possible
3. Show any remaining issues that need manual fixes

### Manual Fixes

For issues that can't be fixed automatically:

1. **Module imports not at top of file**
   - Move all import statements to the beginning of the file, after the docstring
   - Keep them organized: standard library first, then third-party, then local imports

2. **Unused imports**
   - Remove any import statements for modules that aren't used in the code

3. **F-strings without variables**
   - Replace `f"Static text"` with regular strings: `"Static text"`

## GitHub Actions

The project includes a GitHub Actions workflow that runs linting checks automatically on all pull requests and pushes to the main branch.

If linting fails in GitHub Actions, you'll see detailed errors in the action logs and can fix them using the methods described above.

## Pre-commit Hook (Optional)

For a smoother workflow, you can set up a pre-commit hook to check for linting issues before committing:

```bash
pip install pre-commit
pre-commit install
```

Create a `.pre-commit-config.yaml` file in the project root:

```yaml
repos:
-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
    -   id: ruff
        args: [--fix]
```

This will automatically check and fix linting issues before each commit. 