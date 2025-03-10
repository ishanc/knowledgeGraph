#!/bin/bash

# Exit on any error
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Ensure we're using the virtual environment
VIRTUAL_ENV_PATH="venv/bin/activate"
if [ -f "$VIRTUAL_ENV_PATH" ]; then
    source "$VIRTUAL_ENV_PATH"
else
    echo "Virtual environment activation failed!"
    exit 1
fi

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install system dependency for pytesseract
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew install tesseract
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
fi

# Verify Flask installation
python -c "import flask" || {
    echo "Flask installation failed. Please try running: pip install flask"
    exit 1
}

echo "Dependencies installed successfully!"
echo "To run the server, use: source venv/bin/activate && python app.py"
