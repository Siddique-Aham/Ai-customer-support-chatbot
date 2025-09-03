import re
import json
import numpy as np
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra spaces"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def save_json(data: Dict[str, Any], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def format_response(response: str) -> str:
    """Format response to be more natural"""
    response = response.capitalize()
    if not response.endswith(('.', '!', '?')):
        response += '.'
    return response