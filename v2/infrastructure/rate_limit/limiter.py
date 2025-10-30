"""Rate limiting for V2 API.

Distributed rate limiting using Supabase for tracking.
"""

from typing import Optional

from v2.core.logging import logger
from v2.infrastructure.database.repositories import APICallRepository


class RateLimiter:
    """Rate limiter using Supabase for distributed rate limiting."""

    def __init__(self):
        """Initialize rate limiter with API call repository."""
        self._repo = APICallRepository()

    async def check_limit(
        self, user_id: Optional[str], endpoint: str, limit_per_minute: int
    ) -> bool:
        """Check if user/endpoint has exceeded rate limit.

        Args:
            user_id: User UUID (None for anonymous)
            endpoint: Endpoint name
            limit_per_minute: Maximum requests per minute

        Returns:
            True if allowed, False if rate limited
        """
        try:
            # Count recent requests
            count = self._repo.count_recent_calls(
                user_id=user_id, endpoint=endpoint, window_minutes=1
            )

            # Check if limit exceeded
            if count >= limit_per_minute:
                logger.warning(
                    "rate_limit_exceeded",
                    user_id=user_id or "anonymous",
                    endpoint=endpoint,
                    count=count,
                    limit=limit_per_minute,
                )
                return False

            logger.debug(
                "rate_limit_check_passed",
                user_id=user_id or "anonymous",
                endpoint=endpoint,
                count=count,
                limit=limit_per_minute,
            )
            return True

        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e))
            return True  # Fail open (allow request if rate limit check fails)
