"""Tool decorators for V2 API.

Provides standardized error handling and response formatting for all tools.
"""

from datetime import datetime
from typing import Callable


def enrichment_tool(source: str):
    """Decorator that standardizes error handling and response format for enrichment tools.

    Args:
        source: Tool source/name for metadata

    Returns:
        Decorated function that returns standardized response format:
        - success: bool
        - data: tool output (on success)
        - error: error message (on failure)
        - metadata: source and timestamp
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            try:
                data = await func(*args, **kwargs)
                return {
                    "success": True,
                    "data": data,
                    "metadata": {
                        "source": source,
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "metadata": {
                        "source": source,
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                }
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
