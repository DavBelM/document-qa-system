"""
Utility functions for the Document Q&A system.
Provides helper functions for common operations and data processing.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """
    Setup environment variables and configuration.
    """
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("Environment setup completed")


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        model: Model to use for token counting
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback: rough estimation
        return len(text.split()) * 1.3


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
        
        stat = path.stat()
        
        return {
            "name": path.name,
            "size": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "extension": path.suffix,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "is_directory": path.is_dir()
        }
        
    except Exception as e:
        return {"error": str(e)}


def create_directory_if_not_exists(directory_path: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Text to extract keywords from
        num_keywords: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction (in production, use NLP libraries)
    import re
    from collections import Counter
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'throughout',
        'within', 'without', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words and count frequency
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    
    # Return top keywords
    return [word for word, _ in word_counts.most_common(num_keywords)]


def format_response_for_display(response: Dict[str, Any]) -> str:
    """
    Format a QA response for display in UI.
    
    Args:
        response: Response dictionary from QA chain
        
    Returns:
        Formatted string for display
    """
    formatted = f"**Answer:** {response.get('answer', 'No answer provided')}\n\n"
    
    sources = response.get('sources', [])
    if sources:
        formatted += "**Sources:**\n"
        for i, source in enumerate(sources, 1):
            filename = source.get('filename', 'Unknown')
            preview = source.get('content_preview', '')
            formatted += f"{i}. **{filename}** - {preview}\n"
    
    return formatted


def estimate_cost(tokens: int, model: str = "gpt-3.5-turbo") -> float:
    """
    Estimate the cost of API calls based on token count.
    
    Args:
        tokens: Number of tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    # Pricing as of 2024 (update as needed)
    pricing = {
        "gpt-3.5-turbo": 0.0015 / 1000,  # per 1K tokens
        "gpt-4": 0.03 / 1000,
        "text-embedding-ada-002": 0.0001 / 1000
    }
    
    rate = pricing.get(model, 0.002 / 1000)  # default rate
    return tokens * rate


def validate_openai_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if format is valid
    """
    # Basic format validation
    if not api_key:
        return False
    
    # OpenAI API keys start with 'sk-' and have specific length
    if not api_key.startswith('sk-'):
        return False
    
    if len(api_key) < 50:  # Minimum expected length
        return False
    
    return True


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}
