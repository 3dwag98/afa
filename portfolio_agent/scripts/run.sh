#!/bin/bash

# Portfolio Agent Run Script
# Usage: ./scripts/run.sh [install|test|run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
PYTEST="$VENV_DIR/bin/pytest"

case "${1:-run}" in
    install)
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        echo "Installing dependencies..."
        "$PIP" install --upgrade pip
        "$PIP" install -r "$PROJECT_DIR/requirements.txt"
        echo "Installation complete!"
        ;;
    test)
        echo "Running tests..."
        "$PYTEST" "$PROJECT_DIR/tests/"
        ;;
    run)
        if [ ! -d "$VENV_DIR" ]; then
            echo "Virtual environment not found. Running 'install' first..."
            "$0" install
        fi
        echo "Running portfolio agent..."
        "$PYTHON" "$PROJECT_DIR/main.py"
        ;;
    *)
        echo "Usage: $0 {install|test|run}"
        exit 1
        ;;
esac
