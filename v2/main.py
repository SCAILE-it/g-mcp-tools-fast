"""Main entry point for V2 API.

Initializes FastAPI app with TOOLS registry and makes it runnable standalone.

Usage:
    Development: uvicorn v2.main:app --reload --port 8000
    Production: uvicorn v2.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from v2.api import create_app
from v2.tools.registry import get_tools_registry

# Initialize FastAPI app with TOOLS registry
app = create_app(tools_registry=get_tools_registry())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "v2.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
