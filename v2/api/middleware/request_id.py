"""Request ID middleware for V2 API."""

import uuid

from fastapi import Request


async def request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request and response headers."""
    request_id = str(uuid.uuid4())

    # Add to structured logging context if structlog available
    try:
        import structlog
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            response = await call_next(request)
    except ImportError:
        response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    return response
