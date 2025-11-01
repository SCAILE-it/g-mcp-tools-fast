"""JWT authentication for V2 API.

Handles JWT token verification using Supabase JWT secret.
"""

from typing import Optional

import jwt
from fastapi import HTTPException

from v2.config import config
from v2.core.logging import logger


def verify_jwt_token(token: str) -> str:
    """Verify JWT token locally using Supabase JWT secret.

    Args:
        token: JWT token string (without "Bearer " prefix)

    Returns:
        User ID (UUID string) from token payload

    Raises:
        HTTPException: 401 if token invalid or expired, 500 if not configured
    """
    jwt_secret = config.supabase_jwt_secret()

    if not jwt_secret:
        logger.error("jwt_verification_failed", reason="SUPABASE_JWT_SECRET not configured")
        raise HTTPException(
            status_code=500,
            detail="JWT authentication not configured. Contact administrator.",
        )

    try:
        # Decode and verify token
        payload = jwt.decode(
            token, jwt_secret, algorithms=["HS256"], audience="authenticated"
        )

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")

        logger.debug("jwt_token_verified", user_id=user_id)
        return user_id

    except jwt.ExpiredSignatureError:
        logger.warning("jwt_token_expired")
        raise HTTPException(status_code=401, detail="Token expired")

    except jwt.InvalidTokenError as e:
        logger.warning("jwt_token_invalid", error=str(e))
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
