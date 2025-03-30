#!/usr/bin/env python
"""
LTX-Video Linting Fix Script

This script automatically fixes linting issues in Python files using ruff.
It applies common code style fixes for issues like:
- Unused imports
- Module imports not at top of file
- f-strings without variables
- And other auto-fixable issues

Usage:
    python USER_DIR/fix_linting.py

Requirements:
    pip install ruff
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Get the current script directory
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    # Change to project root
    os.chdir(project_root)
    
    # Get all Python files in USER_DIR
    user_dir_python_files = list(script_dir.glob("*.py"))
    
    # Format the list of files for the command
    file_paths = [str(file) for file in user_dir_python_files]
    
    if not file_paths:
        print("No Python files found in USER_DIR.")
        return
    
    print(f"Found {len(file_paths)} Python files to check.")
    print("\nLinting issues before fixes:")
    
    # First run without fixing to show issues
    subprocess.run(["ruff", "check"] + file_paths)
    
    print("\nApplying fixes...")
    
    # Run ruff with the --fix option
    result = subprocess.run(["ruff", "check", "--fix"] + file_paths, capture_output=True, text=True)
    
    # Check if ruff was successful
    if result.returncode != 0:
        print("Some issues couldn't be fixed automatically:")
        print(result.stderr)
    else:
        print("All fixable issues were resolved!")
    
    # Run again to show remaining issues
    print("\nRemaining linting issues (may need manual fixes):")
    subprocess.run(["ruff", "check"] + file_paths)

if __name__ == "__main__":
    # Check if ruff is installed
    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: 'ruff' is not installed or not in the PATH.")
        print("Please install ruff: pip install ruff")
        sys.exit(1)
    
    main() 