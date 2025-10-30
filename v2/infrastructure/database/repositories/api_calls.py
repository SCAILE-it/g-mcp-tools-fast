"""API calls repository for V2 API.

Handles logging and querying API call records.
"""

from datetime import datetime, timedelta
from typing import Optional

from v2.core.logging import logger
from v2.infrastructure.database.models import APICallRecord
from v2.infrastructure.database.repositories.base import BaseRepository


class APICallRepository(BaseRepository[APICallRecord]):
    """Repository for api_calls table operations."""

    def table_name(self) -> str:
        """Return table name."""
        return "api_calls"

    def log_call(self, record: APICallRecord) -> None:
        """Log an API call to the database.

        Args:
            record: API call record to log

        Note:
            Fails silently if Supabase not configured (graceful degradation).
        """
        try:
            if not self._client.is_configured():
                logger.warning("api_call_logging_skipped", reason="Supabase not configured")
                return

            # Convert record to dict for insertion
            data = {
                "user_id": record.user_id,
                "tool_name": record.tool_name,
                "tool_type": record.tool_type,
                "input_data": record.input_data,
                "output_data": record.output_data,
                "success": record.success,
                "processing_ms": record.processing_ms,
                "error_message": record.error_message,
                "tokens_used": record.tokens_used,
            }

            self.db.table(self.table_name()).insert(data).execute()

            logger.info(
                "api_call_logged",
                user_id=record.user_id,
                tool_name=record.tool_name,
                success=record.success,
                processing_ms=record.processing_ms,
            )

        except Exception as e:
            logger.warning(
                "api_call_logging_failed",
                error=str(e),
                tool_name=record.tool_name,
            )

    def count_recent_calls(
        self, user_id: Optional[str], endpoint: str, window_minutes: int = 1
    ) -> int:
        """Count API calls in recent time window for rate limiting.

        Args:
            user_id: User UUID (or fixed UUID for anonymous)
            endpoint: Endpoint name
            window_minutes: Time window in minutes

        Returns:
            Number of calls in the time window
        """
        try:
            if not self._client.is_configured():
                logger.warning("rate_limit_check_skipped", reason="Supabase not configured")
                return 0

            # Use fixed UUID for anonymous users (Supabase user_id column is UUID type)
            key = user_id if user_id else "00000000-0000-0000-0000-000000000000"
            now = datetime.now()
            window_start = now - timedelta(minutes=window_minutes)

            # Count requests in the time window
            result = (
                self.db.table(self.table_name())
                .select("id", count="exact")
                .eq("user_id", key)
                .eq("tool_name", endpoint)
                .gte("created_at", window_start.isoformat())
                .execute()
            )

            count = result.count if hasattr(result, "count") else 0
            return count

        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e))
            return 0  # Fail open (allow request if check fails)
