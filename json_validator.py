import json
from jsonschema import validate, ValidationError
import logging
import math

logger = logging.getLogger(__name__)

# Define the expected JSON schema
JSON_SCHEMA = {
    "type": "object",
    "required": ["filename", "file_type", "data", "timestamp"],
    "properties": {
        "filename": {"type": "string"},
        "file_type": {"type": "string"},
        "timestamp": {"type": ["string", "null"]},
        "data": {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array"},
                        {"type": "object"}
                    ]
                }
            }
        }
    }
}

def validate_json_structure(json_data):
    try:
        validate(instance=json_data, schema=JSON_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)

def check_json_content(data):
    """Perform detailed content validation"""
    try:
        def check_value(value):
            if isinstance(value, dict):
                return all(check_value(v) for v in value.values())
            elif isinstance(value, list):
                return all(check_value(v) for v in value)
            elif isinstance(value, str):
                return True
            elif value is None:
                return True
            elif isinstance(value, (int, float, bool)):
                return True
            else:
                return False

        return check_value(data), None
    except Exception as e:
        return False, f"Content validation error: {str(e)}"

def validate_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for common JSON syntax issues
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                line_no = e.lineno
                col_no = e.colno
                lines = content.split('\n')
                context = '\n'.join(lines[max(0, line_no-2):min(line_no+1, len(lines))])
                error_details = f"JSON syntax error at line {line_no}, column {col_no}:\n"
                error_details += f"Context:\n{context}\n"
                error_details += f"Error: {str(e)}"
                return False, error_details

            # Schema validation
            schema_valid, schema_error = validate_json_structure(data)
            if not schema_valid:
                return False, f"Schema validation error: {schema_error}"

            # Content validation
            content_valid, content_error = check_json_content(data)
            if not content_valid:
                return False, f"Content validation error: {content_error}"

            return True, None
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def clean_problematic_value(value):
    """Clean any problematic values that might cause JSON issues"""
    if isinstance(value, dict):
        return {k: clean_problematic_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_problematic_value(v) for v in value]
    elif isinstance(value, (int, float)):
        # Check for NaN without pandas dependency
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    elif isinstance(value, str):
        # Remove or replace problematic characters
        value = value.replace('\x00', '')  # Remove null bytes
        value = ''.join(char for char in value if ord(char) >= 32 or char == '\n')
        return value
    elif value is None:
        return None
    else:
        return str(value)

def fix_json_content(content):
    """Try to fix common JSON issues"""
    try:
        if isinstance(content, str):
            try:
                # Try to parse the string as JSON first
                data = json.loads(content)
            except json.JSONDecodeError:
                # If parsing fails, try to clean the string
                content = content.replace('\r\n', '\n').replace('\r', '\n')
                content = content.replace('\x00', '')
                content = ''.join(char for char in content if ord(char) >= 32 or char == '\n')
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Could not parse JSON content even after cleaning: {str(e)}")
                    return content
        else:
            data = content
        
        # Clean all values in the parsed data
        cleaned_data = clean_problematic_value(data)
        
        # Serialize back to JSON with proper formatting
        return json.dumps(cleaned_data, indent=4, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Error fixing JSON: {str(e)}")
        return content
