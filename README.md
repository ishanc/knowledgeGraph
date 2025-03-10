# Knowledge Graph Document Parser

This project provides utilities to parse various document formats (PDF, DOC, DOCX, TXT, RTF, CSV, XLS, XLSX, and images) and convert them into JSON format for further processing.

## Prerequisites

- Python 3.6 or higher
- For macOS users: Homebrew
- For Linux users: apt package manager
- Sufficient permissions to install system packages

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd knowledgeGraph
```

2. Run the installation script:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

This will:
- Create a Python virtual environment
- Install all required Python packages
- Install Tesseract OCR system dependency

## Supported File Types

- Text files (.txt, .rtf)
- Spreadsheets (.csv, .xls, .xlsx)
- Documents (.doc, .docx)
- PDFs (.pdf)
- Images (.png, .jpg, .jpeg)

## Usage

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Use the script by importing and calling the process_file function:
```python
from knowledgeGraph import process_file

result = process_file("path/to/your/document")
print(result)
```

The script will:
- Read the input file
- Extract content based on file type
- Generate a JSON file with the same name as the input file
- Return a success/error message

## Output Format

The generated JSON files will have the following structure:
```json
{
    "filename": "input_file_name",
    "file_type": ".extension",
    "data": {
        "content": [extracted_content]
    }
}
```

## Troubleshooting

If you encounter OCR-related issues:
- Ensure Tesseract is properly installed
- For macOS: `brew install tesseract`
- For Linux: `sudo apt-get install tesseract-ocr`

## License

[Add your license information here]
