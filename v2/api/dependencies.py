"""FastAPI dependencies for V2 API.

Dependency injection functions for route handlers.
"""

from typing import Any, Dict

from fastapi import Request


def get_tools_registry(request: Request) -> Dict[str, Any]:
    """Get TOOLS registry from app state.

    Args:
        request: FastAPI request object

    Returns:
        Dict of tool name to tool config/function

    Note:
        Returns empty dict if tools_registry not set in app.state.
        Set via: app.state.tools_registry = {...}
    """
    return getattr(request.app.state, "tools_registry", {})
