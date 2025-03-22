import re
import html

def sanitize_input(input_text):
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_text (str): The input text to sanitize.
        
    Returns:
        str: The sanitized text.
    """
    # Escape HTML entities
    escaped_text = html.escape(input_text)
    
    # Additional sanitization can be added here as needed
    
    return escaped_text

def extract_urls(text):
    """
    Extract URLs from a text.
    
    Args:
        text (str): The text to extract URLs from.
        
    Returns:
        list: A list of extracted URLs.
    """
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)
