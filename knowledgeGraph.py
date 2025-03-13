import pandas as pd
import json
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract
import os
import re
import logging
from json_validator import fix_json_content

logger = logging.getLogger(__name__)

def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and extra whitespace
        text = re.sub(r'[\n\r\t]+', ' ', text)  # Replace newlines, tabs with space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()  # Remove leading/trailing whitespace
    return text

def clean_content(content):
    if isinstance(content, list):
        return [clean_text(item) if isinstance(item, str) else item for item in content]
    elif isinstance(content, dict):
        return {k: clean_text(v) if isinstance(v, str) else v for k, v in content.items()}
    else:
        return clean_text(content)

def clean_value(value):
    """Clean individual values to ensure JSON compatibility"""
    if pd.isna(value):  # Handle NaN and None values
        return None
    elif isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return value
    elif isinstance(value, str):
        return clean_text(value)
    else:
        return str(value)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return {'content': clean_text(content)}

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    # Convert all values to JSON-compatible format
    records = []
    for record in df.to_dict(orient='records'):
        cleaned_record = {k: clean_value(v) for k, v in record.items()}
        records.append(cleaned_record)
    return {'content': records}

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    content = df.to_dict(orient='records')
    return {'content': clean_content(content)}

def read_doc_file(file_path):
    doc = Document(file_path)
    content = [paragraph.text for paragraph in doc.paragraphs]
    return {'content': clean_content(content)}

def read_pdf_file(file_path):
    content = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            content.append(page.extract_text())
    return {'content': clean_content(content)}

def read_image_file(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return {'content': clean_text(text)}

def process_file(file_path, output_folder='processed_files', timestamp=None):
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.debug(f"Processing file: {file_path} with extension {file_extension}")
        
        handlers = {
            '.txt': read_text_file,
            '.rtf': read_text_file,
            '.csv': read_csv_file,
            '.xls': read_excel_file,
            '.xlsx': read_excel_file,
            '.doc': read_doc_file,
            '.docx': read_doc_file,
            '.pdf': read_pdf_file,
            '.png': read_image_file,
            '.jpg': read_image_file,
            '.jpeg': read_image_file
        }
        
        if file_extension in handlers:
            try:
                content = handlers[file_extension](file_path)
                filename = os.path.basename(file_path)
                result = {
                    'filename': filename,
                    'file_type': file_extension,
                    'data': content,
                    'timestamp': timestamp
                }
                
                # Ensure output directory exists
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
                
                try:
                    # First, ensure all values are JSON serializable
                    json_str = json.dumps(result, indent=4, ensure_ascii=False, default=str)
                    # Parse it back to validate structure
                    json.loads(json_str)
                    # Write the validated JSON
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(json_str)
                except Exception as je:
                    logger.error(f"JSON serialization error: {str(je)}")
                    raise
                
                logger.debug(f"Successfully processed {filename}")
                return f"Successfully processed {filename}"
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return f"Error processing {file_path}: {str(e)}"
        else:
            return f"Unsupported file type: {file_extension}"
    except Exception as e:
        logger.error(f"Error in process_file: {str(e)}")
        return f"Error processing file: {str(e)}"

# Example usage
if __name__ == "__main__":
    file_path = "path/to/your/file"  # Replace with actual file path
    print(process_file(file_path))
