"""Scheduling utilities for V2 API.

Helpers for calculating next run times for scheduled jobs.
"""

from datetime import datetime, timedelta
from typing import Optional


def calculate_next_run_at(
    schedule_preset: str, from_time: Optional[datetime] = None
) -> datetime:
    """Calculate next scheduled run time based on preset.

    Args:
        schedule_preset: 'daily', 'weekly', or 'monthly'
        from_time: Starting time (defaults to now)

    Returns:
        Next scheduled run time

    Raises:
        ValueError: If invalid preset

    Example:
        >>> next_run = calculate_next_run_at("daily")
        >>> # Returns datetime 24 hours from now
        >>> next_run = calculate_next_run_at("weekly", datetime(2025, 10, 30))
        >>> # Returns datetime(2025, 11, 6)
    """
    if from_time is None:
        from_time = datetime.now()

    if schedule_preset == "daily":
        return from_time + timedelta(days=1)
    elif schedule_preset == "weekly":
        return from_time + timedelta(days=7)
    elif schedule_preset == "monthly":
        return from_time + timedelta(days=30)
    else:
        raise ValueError(
            f"Invalid schedule_preset: {schedule_preset}. "
            f"Must be 'daily', 'weekly', or 'monthly'"
        )
