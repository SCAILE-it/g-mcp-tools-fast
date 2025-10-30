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
    """Represents a saved/scheduled job in saved_queries table."""

    id: str
    user_id: str
    workflow_name: str
    workflow_type: str
    params: Dict[str, Any]
    is_scheduled: bool
    schedule_preset: Optional[str] = None
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
