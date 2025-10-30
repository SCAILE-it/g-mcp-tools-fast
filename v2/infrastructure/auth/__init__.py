"""Authentication module for V2 API.

Provides JWT and legacy API key authentication.

Fixes V1 issue: Auth functions nested inside api() function (SRP violation).
Now properly separated into dedicated auth module.
"""

from v2.infrastructure.auth.api_key import verify_api_key
from v2.infrastructure.auth.deps import get_current_user
from v2.infrastructure.auth.jwt import verify_jwt_token

__all__ = [
    "verify_jwt_token",
    "verify_api_key",
    "get_current_user",
]
