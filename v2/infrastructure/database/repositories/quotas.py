"""User quota repository for V2 API.

Handles quota checking and enforcement.
"""

from typing import Optional

from fastapi import HTTPException

from v2.core.logging import logger
from v2.infrastructure.database.repositories.base import BaseRepository


class QuotaRepository(BaseRepository[None]):
    """Repository for user quota operations."""

    def table_name(self) -> str:
        """Not applicable - uses RPC function."""
        return "user_quotas"

    def check_quota(self, user_id: str) -> None:
        """Check if user has available quota.

        Args:
            user_id: User UUID

        Raises:
            HTTPException: 429 if quota exceeded

        Note:
            Uses Supabase RPC function check_user_quota which handles:
            - Monthly quota tracking
            - Usage increment
            - Automatic reset on new month
        """
        try:
            if not self._client.is_configured():
                logger.warning("quota_check_skipped", reason="Supabase not configured")
                return

            # Call stored procedure
            result = self.db.rpc("check_user_quota", {"p_user_id": user_id}).execute()

            # RPC returns true if quota available, raises exception if exceeded
            if not result.data:
                logger.warning("quota_exceeded", user_id=user_id)
                raise HTTPException(
                    status_code=429,
                    detail="Monthly API quota exceeded. Upgrade your plan or wait for reset.",
                )

            logger.debug("quota_check_passed", user_id=user_id)

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Log error but fail open (allow request if quota check fails)
            logger.error("quota_check_failed", user_id=user_id, error=str(e))

    def get_usage(self, user_id: str) -> Optional[dict]:
        """Get current quota usage for a user.

        Args:
            user_id: User UUID

        Returns:
            Dict with usage, limit, remaining, or None if not configured
        """
        try:
            if not self._client.is_configured():
                return None

            result = (
                self.db.table(self.table_name())
                .select("monthly_quota_limit, monthly_usage, quota_reset_date")
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                return None

            data = result.data[0]
            return {
                "limit": data.get("monthly_quota_limit", 0),
                "usage": data.get("monthly_usage", 0),
                "remaining": data.get("monthly_quota_limit", 0) - data.get("monthly_usage", 0),
                "reset_date": data.get("quota_reset_date"),
            }

        except Exception as e:
            logger.error("quota_usage_fetch_failed", user_id=user_id, error=str(e))
            return None
