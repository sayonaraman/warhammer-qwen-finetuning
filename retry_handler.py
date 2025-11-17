"""
Universal Retry Handler with Exponential Backoff

Provides decorator for automatic retry with progressive backoff delays.
Handles transient errors: network issues, rate limits, temporary API failures.
"""

import time
import sys
from functools import wraps
from requests.exceptions import RequestException, Timeout, ConnectionError


def retry_with_backoff(
    max_retries=3,
    base_delay=5,
    max_delay=30,
    operation_name="Operation"
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 5)
        max_delay: Maximum delay cap in seconds (default: 30)
        operation_name: Human-readable operation name for logs
        
    Returns:
        Decorated function with retry logic
        
    Usage:
        @retry_with_backoff(operation_name="API Call")
        def my_api_call():
            # Your code here
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    retries += 1
                    
                    # If we've exhausted all retries, raise the error
                    if retries > max_retries:
                        print(f"[ERROR] {operation_name} failed after {max_retries} retries")
                        sys.stdout.flush()
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    
                    print(f"[RETRY] {operation_name} failed (attempt {retries}/{max_retries})")
                    print(f"        Error: {str(e)}")
                    print(f"        Retrying in {delay}s...")
                    sys.stdout.flush()
                    
                    time.sleep(delay)
            
        return wrapper
    return decorator

