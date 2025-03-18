import json
import logging
import time
from typing import Optional, Any
import requests

logger = logging.getLogger(__name__)

def validate_json_structure(text: str) -> bool:
    """Validate if the text is a properly formatted JSON."""
    text = text.strip()
    return text.startswith('{') and text.endswith('}')

def clean_llm_response(response: str) -> str:
    """Clean LLM response to ensure valid JSON."""
    # Find the first { and last }
    start = response.find('{')
    end = response.rfind('}')
    
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", response, 0)
    
    return response[start:end + 1]

def validate_llm_response(response: Any) -> Optional[Any]:
    """Validate and clean LLM response."""
    try:
        # Handle pre-parsed JSON
        if isinstance(response, dict):
            return response
            
        # Handle string responses
        if isinstance(response, str):
            # Clean and validate JSON structure
            if not validate_json_structure(response):
                response = clean_llm_response(response)
            return json.loads(response)
            
        raise ValueError(f"Unexpected response type: {type(response)}")
        
    except Exception as e:
        logger.error(f"Error validating LLM response: {str(e)}")
        return None

def retry_llm_call(func, *args, max_retries=3, **kwargs) -> Optional[Any]:
    """Retry LLM calls with JSON validation."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = func(*args, **kwargs)
            validated_response = validate_llm_response(response)
            
            if validated_response is not None:
                return validated_response
            
            raise ValueError("Failed to validate response")
            
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout, 
                requests.exceptions.RequestException) as e:
            last_error = e
            logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Longer delay for connection issues
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                
    logger.error(f"All attempts failed. Last error: {str(last_error)}")
    return None

def format_llm_prompt(prompt: str, include_format_instructions: bool = True) -> str:
    """Format prompt with JSON requirements."""
    if include_format_instructions:
        return f"""{prompt}

{FORMAT_INSTRUCTIONS}

Example response:
{json.dumps(EXAMPLE_RESPONSE, indent=2)}

Ensure your response is a valid JSON object following this format exactly."""
    return prompt

# Example JSON schema for validation
LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["text", "category"]
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "type": {"type": "string"},
                    "target": {"type": "string"},
                    "weight": {"type": "number"}
                },
                "required": ["source", "type", "target"]
            }
        }
    }
}

# Example prompt template
EXAMPLE_RESPONSE = {
    "entities": [
        {
            "text": "Enterprise Architecture",
            "category": "CONCEPT",
            "confidence": 0.95
        }
    ],
    "relationships": [
        {
            "source": "Enterprise Architecture",
            "type": "includes",
            "target": "Business Architecture",
            "weight": 0.8
        }
    ]
}

FORMAT_INSTRUCTIONS = """
Your response must be a valid JSON object with the following structure:
{
    "entities": [
        {
            "text": "string",
            "category": "string",
            "confidence": number
        }
    ],
    "relationships": [
        {
            "source": "string",
            "type": "string",
            "target": "string",
            "weight": number
        }
    ]
}

Requirements:
1. Must be a complete JSON object (starts with { and ends with })
2. No content outside the JSON object
3. All strings must be double-quoted
4. All properties must be quoted
5. Numbers should not be quoted
6. Must contain either entities or relationships array or both
"""
