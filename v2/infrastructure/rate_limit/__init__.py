"""Rate limiting module for V2 API.

Provides distributed rate limiting using Supabase for tracking.

Fixes V1 issue: Rate limiting nested inside api() function (SRP violation).
Now properly separated with decorator pattern for easy route protection.
"""

from v2.infrastructure.rate_limit.deps import check_rate_limit, enforce_rate_limit
from v2.infrastructure.rate_limit.limiter import RateLimiter

__all__ = [
    "RateLimiter",
    "check_rate_limit",
    "enforce_rate_limit",
]
