"""Rate limiting middleware for V2 API."""

import os
from datetime import datetime, timedelta
from typing import Optional

from v2.core.logging import logger


async def check_rate_limit(user_id: Optional[str], endpoint: str, limit_per_minute: int) -> bool:
    """Check if user/endpoint exceeded rate limit. Sliding window, fail-open on errors."""
    try:
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("rate_limit_check_skipped", reason="Supabase not configured")
            return True  # Allow if DB not configured

        supabase = create_client(supabase_url, supabase_key)

        # Use fixed UUID for anonymous users (Supabase user_id column is UUID type)
        key = user_id if user_id else "00000000-0000-0000-0000-000000000000"
        now = datetime.now()
        window_start = now - timedelta(minutes=1)

        # Count requests in the last minute
        result = (
            supabase.table("api_calls")
            .select("id", count="exact")
            .eq("user_id", key)
            .eq("tool_name", endpoint)
            .gte("created_at", window_start.isoformat())
            .execute()
        )

        count = result.count if hasattr(result, "count") else 0

        if count >= limit_per_minute:
            logger.warning(
                "rate_limit_exceeded",
                user_id=key,
                endpoint=endpoint,
                count=count,
                limit=limit_per_minute,
            )
            return False

        return True

    except Exception as e:
        logger.error("rate_limit_check_failed", error=str(e))
        return True  # Fail open (allow request if rate limit check fails)
