import pandas as pd
import json
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract
import os

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return {'content': file.read()}

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return {'content': df.to_dict(orient='records')}

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return {'content': df.to_dict(orient='records')}

def read_doc_file(file_path):
    doc = Document(file_path)
    content = [paragraph.text for paragraph in doc.paragraphs]
    return {'content': content}

def read_pdf_file(file_path):
    content = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            content.append(page.extract_text())
    return {'content': content}

def read_image_file(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return {'content': text}

def process_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
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
            result = {
                'filename': os.path.basename(file_path),
                'file_type': file_extension,
                'data': content
            }
            
            # Save to JSON
            output_file = f"{os.path.splitext(file_path)[0]}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            return f"Successfully processed {file_path}"
        except Exception as e:
            return f"Error processing {file_path}: {str(e)}"
    else:
        return f"Unsupported file type: {file_extension}"

# Example usage
if __name__ == "__main__":
    file_path = "path/to/your/file"  # Replace with actual file path
    print(process_file(file_path))
