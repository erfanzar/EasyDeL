#!/bin/bash
# Script to set up pre-commit hooks for EasyDeL

echo "Setting up pre-commit hooks for EasyDeL..."

# Check if we're in a virtual environment, otherwise use the project venv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating .venv..."
        source .venv/bin/activate
    elif [ -d "/home/erfan/Projects/EasyDeL/.venv" ]; then
        echo "Activating EasyDeL venv..."
        source /home/erfan/Projects/EasyDeL/.venv/bin/activate
    fi
fi

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit with uv..."
    uv pip install pre-commit
fi

# Install the pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files (optional, can be slow on first run)
read -p "Do you want to run pre-commit on all files now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running pre-commit on all files..."
    pre-commit run --all-files
fi

echo "Pre-commit setup complete!"
echo ""
echo "Pre-commit will now run automatically on git commit."
echo "To run manually: pre-commit run --all-files"
echo "To update hooks: pre-commit autoupdate"
