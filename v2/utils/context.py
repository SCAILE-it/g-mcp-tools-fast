"""Request context utilities for V2 API.

Extracts system context from FastAPI requests.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import Request


def get_system_context(request: Request) -> Dict[str, Any]:
    """Extract system context from request headers.

    Args:
        request: FastAPI Request object

    Returns:
        Dict with date, country, timezone, language

    Example:
        >>> context = get_system_context(request)
        >>> # {"date": "2025-10-30", "datetime": "2025-10-30T12:00:00",
        >>> #  "country": "US", "timezone": "UTC", "language": "en"}
    """
    return {
        "date": datetime.now().isoformat()[:10],  # YYYY-MM-DD
        "datetime": datetime.now().isoformat(),
        "country": request.headers.get("cf-ipcountry", "US"),
        "timezone": request.headers.get("timezone", "UTC"),
        "language": (
            request.headers.get("accept-language", "en")[:2]
            if request.headers.get("accept-language")
            else "en"
        ),
    }
