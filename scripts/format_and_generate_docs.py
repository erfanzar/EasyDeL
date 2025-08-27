#!/usr/bin/env python3
"""
Standalone script to format code and generate API documentation.
This is NOT a pre-commit hook - use .pre-commit-config.yaml instead.
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def format_code(directory="easydel"):
    """Format Python code using ruff."""
    print(f"Formatting code in {directory}...")

    python_files = list(Path(directory).rglob("*.py"))

    if not python_files:
        print("No Python files found.")
        return

    # Run ruff check with fixes
    print("Running ruff check...")
    subprocess.run(
        ["ruff", "check", "--fix", "--config", "pyproject.toml"] + [str(f) for f in python_files], check=False
    )

    # Run ruff format
    print("Running ruff format...")
    subprocess.run(["ruff", "format", "--config", "pyproject.toml"] + [str(f) for f in python_files], check=False)

    print("Code formatting complete!")


def generate_api_docs():
    """Generate API documentation RST files."""
    print("Generating API documentation...")

    # Import the existing doc generation logic
    from pre_commit import main as generate_docs

    generate_docs()

    print("API documentation generated!")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Format code and generate documentation")
    parser.add_argument("--format", action="store_true", help="Format code with ruff")
    parser.add_argument("--docs", action="store_true", help="Generate API documentation")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--directory", default="easydel", help="Directory to format (default: easydel)")

    args = parser.parse_args()

    # Default to running all if no specific flags
    if not any([args.format, args.docs, args.all]):
        args.all = True

    if args.all or args.format:
        format_code(args.directory)

    if args.all or args.docs:
        generate_api_docs()


if __name__ == "__main__":
    main()
