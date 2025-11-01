"""FastAPI rate limiting dependencies for V2 API.

Provides reusable rate limiting decorators for route protection.
"""

from typing import Optional

from fastapi import HTTPException

from v2.infrastructure.rate_limit.limiter import RateLimiter

# Module-level singleton
_limiter = RateLimiter()


async def check_rate_limit(
    user_id: Optional[str], endpoint: str, limit_per_minute: int
) -> bool:
    """Check rate limit for endpoint.

    Args:
        user_id: User UUID (None for anonymous)
        endpoint: Endpoint name
        limit_per_minute: Maximum requests per minute

    Returns:
        True if allowed

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    allowed = await _limiter.check_limit(user_id, endpoint, limit_per_minute)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {limit_per_minute} requests per minute.",
            headers={"Retry-After": "60"},
        )

    return True


async def enforce_rate_limit(user_id: Optional[str], endpoint: str, limit: int = 60):
    """FastAPI dependency for rate limiting.

    Usage:
        @app.post("/endpoint")
        async def endpoint(
            user_id: str = Depends(get_current_user),
            _: bool = Depends(lambda u=user_id: enforce_rate_limit(u, "/endpoint", 10))
        ):
            ...

    Args:
        user_id: User UUID from auth dependency
        endpoint: Endpoint path
        limit: Requests per minute (default: 60)

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    await check_rate_limit(user_id, endpoint, limit)
