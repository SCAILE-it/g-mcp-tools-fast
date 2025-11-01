"""Saved jobs-related Pydantic models for V2 API.

These models handle creating and scheduling saved jobs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SavedJobCreateRequest(BaseModel):
    """Request for POST /jobs/saved endpoint - create saved job."""
    name: str = Field(
        ...,
        min_length=1,
        description="Job name"
    )
    description: Optional[str] = Field(
        None,
        description="Optional job description"
    )
    tool_name: str = Field(
        ...,
        description="Tool to execute (e.g., 'phone-validation', 'email-intel')"
    )
    params: Dict[str, Any] = Field(
        ...,
        description="Tool-specific parameters",
    )
    is_template: bool = Field(
        default=False,
        description="Whether this is a template job with variables"
    )
    template_vars: Optional[List[str]] = Field(
        None,
        description="List of template variable names (if is_template=true)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Daily Phone Validation",
                    "description": "Validate customer phone numbers",
                    "tool_name": "phone-validation",
                    "params": {"phone_number": "+14155552671"},
                    "is_template": False
                }
            ]
        }
    }


class JobScheduleUpdateRequest(BaseModel):
    """Request for PATCH /jobs/saved/{job_id}/schedule endpoint - update job schedule."""
    is_scheduled: bool = Field(
        ...,
        description="Enable (true) or disable (false) scheduling"
    )
    schedule_preset: Optional[str] = Field(
        None,
        description="Schedule preset: 'daily', 'weekly', or 'monthly'",
        pattern="^(daily|weekly|monthly)$"
    )
    schedule_cron: Optional[str] = Field(
        None,
        description="Custom cron expression (mutually exclusive with schedule_preset)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_scheduled": True,
                    "schedule_preset": "daily"
                }
            ]
        }
    }
