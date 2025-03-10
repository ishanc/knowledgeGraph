#!/bin/bash

# Initialize git repository
git init

# Add all files
git add .gitignore
git add *.py
git add *.sh
git add requirements.txt
git add README.md
git add templates/
git add static/

# Initial commit
git commit -m "Initial commit: Knowledge Graph Document Parser

- Document processing functionality
- Web interface for file uploads
- Multiple file format support
- JSON output generation"

echo "Git repository initialized successfully!"
echo "Next steps:"
echo "1. Add remote repository: git remote add origin <repository-url>"
echo "2. Push to remote: git push -u origin main"
