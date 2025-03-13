#!/bin/bash

# Exit on any error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Check Python version and install Python 3.10 if needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v python3.10 &> /dev/null; then
        echo "Installing Python 3.10..."
        brew install python@3.10
    fi
    PYTHON_CMD=python3.10
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! command -v python3.10 &> /dev/null; then
        echo "Installing Python 3.10..."
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv
    fi
    PYTHON_CMD=python3.10
else
    echo "Unsupported operating system"
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD --version

# Get Python installation path
PYTHON_PATH=$(which $PYTHON_CMD)
echo "Python path: $PYTHON_PATH"

# Remove existing virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$SCRIPT_DIR/venv"
fi

echo "Creating new virtual environment..."
cd "$SCRIPT_DIR"
$PYTHON_PATH -m venv venv || {
    echo "Failed to create virtual environment using $PYTHON_PATH"
    echo "Trying with python3..."
    python3 -m venv venv || {
        echo "Failed to create virtual environment with python3"
        exit 1
    }
}

# Verify virtual environment creation
if [ ! -f "$SCRIPT_DIR/venv/bin/python" ]; then
    echo "Python executable not found in virtual environment"
    echo "Contents of venv directory:"
    ls -la "$SCRIPT_DIR/venv"
    echo "Contents of venv/bin directory:"
    ls -la "$SCRIPT_DIR/venv/bin"
    exit 1
fi

# Ensure we're using the virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment activation failed!"
    exit 1
fi

echo "Virtual environment activated successfully at: $VIRTUAL_ENV"

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install requirements one by one with error handling
while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    # Skip comments and empty lines
    [[ $requirement =~ ^#.*$ ]] && continue
    [[ -z "$requirement" ]] && continue
    
    echo "Installing $requirement..."
    pip install "$requirement" || {
        echo "Failed to install $requirement"
        exit 1
    }
done < requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm || {
    echo "Failed to download spaCy language model"
    exit 1
}

# Install system dependency for pytesseract
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install tesseract
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install -y tesseract-ocr
fi

# Verify installations
python -c "import spacy; spacy.load('en_core_web_sm')" || {
    echo "spaCy installation verification failed"
    exit 1
}

echo "Dependencies installed successfully!"
echo "To run the server, use: source venv/bin/activate && python app.py"
