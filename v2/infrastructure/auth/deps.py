"""FastAPI authentication dependencies for V2 API.

Provides reusable authentication dependencies for route protection.
"""

from typing import Optional

from fastapi import Header, HTTPException

from v2.config import config
from v2.infrastructure.auth.api_key import verify_api_key
from v2.infrastructure.auth.jwt import verify_jwt_token


async def get_current_user(
    authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)
) -> Optional[str]:
    """Extract user_id from JWT (preferred) or fall back to legacy API key.

    Priority order:
    1. JWT in Authorization header (per-user tracking)
    2. Legacy API key in x-api-key header (no user tracking)
    3. Anonymous (if auth disabled)

    Args:
        authorization: Authorization header with Bearer token
        x_api_key: Legacy API key header

    Returns:
        User UUID if JWT auth, None if API key or anonymous

    Raises:
        HTTPException: 401 if authentication required but not provided/invalid
    """
    # Priority 1: JWT authentication (per-user)
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Use: Authorization: Bearer <token>",
            )

        token = authorization.replace("Bearer ", "")
        return verify_jwt_token(token)  # Returns user_id or raises HTTPException

    # Priority 2: Legacy API key (backward compatible, no user tracking)
    if x_api_key:
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return None  # Valid API key but no user_id

    # Priority 3: Anonymous (only if MODAL_API_KEY not set)
    if not config.modal_api_key():
        return None  # Anonymous access allowed

    # Auth required but not provided
    raise HTTPException(
        status_code=401, detail="Authentication required. Provide JWT token or API key."
    )
