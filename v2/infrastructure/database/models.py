"""Database models for V2 API.

Type-safe dataclasses representing database records.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class APICallRecord:
    """Represents a record in the api_calls table."""

    user_id: Optional[str]
    tool_name: str
    tool_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    processing_ms: int
    error_message: Optional[str] = None
    tokens_used: int = 0
    created_at: Optional[datetime] = None


@dataclass
class SavedJob:
    """Represents a saved/scheduled job in saved_queries table.

    Tool-based saved jobs that can be executed manually or on a schedule.
    """

    id: str
    user_id: str
    name: str
    tool_name: str
    params: Dict[str, Any]
    description: Optional[str] = None
    is_template: bool = False
    template_vars: Optional[list] = None
    is_scheduled: bool = False
    schedule_preset: Optional[str] = None
    schedule_cron: Optional[str] = None
    next_run_at: Optional[str] = None
    last_run_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
