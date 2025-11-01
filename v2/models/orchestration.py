"""Orchestration-related Pydantic models for V2 API.

These models handle AI-powered planning and orchestration requests.
"""

from typing import Optional
from pydantic import Field
from v2.models.base import BaseUserRequestModel


class PlanRequest(BaseUserRequestModel):
    """Request for /plan endpoint - AI-powered execution planning."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"user_request": "Research Tesla and find contact information for their sales team"},
                {"user_request": "Analyze the tech stack for stripe.com and check their domain registration"}
            ]
        }
    }


class OrchestrateRequest(BaseUserRequestModel):
    """Request for /orchestrate endpoint - full AI workflow orchestration."""
    stream: bool = Field(default=True, description="Enable SSE streaming for real-time progress updates")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_request": "Research Tesla and find contact information for their sales team",
                    "stream": True
                }
            ]
        }
    }


class WorkflowGenerateRequest(BaseUserRequestModel):
    """Request for /workflow/generate endpoint - AI workflow generation."""
    save_as: Optional[str] = Field(None, description="Optional workflow name to save after generation")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"user_request": "Create a workflow to validate emails and check tech stack for each domain"}
            ]
        }
    }
