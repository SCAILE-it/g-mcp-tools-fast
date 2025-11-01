"""Quota enforcement middleware for V2 API.

Provides reusable quota checking for FastAPI routes.
"""

from typing import Optional

from v2.infrastructure.database.repositories import QuotaRepository


class QuotaMiddleware:
    """Middleware for enforcing user API quotas."""

    def __init__(self, quota_repository: Optional[QuotaRepository] = None):
        """Initialize quota middleware.

        Args:
            quota_repository: QuotaRepository instance (optional)
        """
        self.quota_repo = quota_repository or QuotaRepository()

    async def check_quota(self, user_id: Optional[str]) -> None:
        """Check if user has available quota.

        Args:
            user_id: User UUID (None for anonymous/API key users)

        Raises:
            HTTPException: 429 if quota exceeded

        Note:
            Only enforces quota for authenticated JWT users.
            Anonymous and API key users bypass quota checks.
        """
        if user_id:
            # Authenticated user - enforce quota
            self.quota_repo.check_quota(user_id)
