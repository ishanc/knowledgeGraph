#!/bin/bash

# Create necessary directories
mkdir -p templates
mkdir -p static
mkdir -p uploads

# Move index.html to templates if it exists in root
if [ -f "index.html" ]; then
    mv index.html templates/
fi

# Create a simple test file to verify static serving
echo "/* Add your custom styles here */" > static/style.css

echo "Project structure setup complete!"
echo "Please ensure index.html is in the templates directory"
ls -R
