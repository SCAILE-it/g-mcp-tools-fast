"""Request models for V2 API."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """Browser automation action types for scraping."""

    CLICK = "click"
    SCROLL = "scroll"
    WAIT = "wait"
    TYPE = "type"
    SCREENSHOT = "screenshot"


class ScrapeAction(BaseModel):
    """Browser action for scraping."""

    type: ActionType
    selector: Optional[str] = None
    text: Optional[str] = None
    milliseconds: Optional[int] = None
    pixels: Optional[int] = None


class ScrapeRequest(BaseModel):
    """Request for /scrape endpoint."""

    url: str
    prompt: str = Field(..., min_length=1)
    output_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")
    actions: Optional[List[ScrapeAction]] = None
    max_pages: Optional[int] = Field(1, ge=1, le=50)
    timeout: Optional[int] = Field(30, ge=5, le=120)
    extract_links: Optional[bool] = False
    use_context_analysis: Optional[bool] = True
    auto_discover_pages: Optional[bool] = False

    class Config:
        populate_by_name = True

    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Ensure URL has scheme."""
        if not v.startswith(("http://", "https://")):
            v = f"https://{v}"
        return v.strip()


class ExecuteRequest(BaseModel):
    """Request for /execute endpoint - single-tool batch processing with SSE."""

    executionId: str
    tool: str
    data: List[Dict[str, Any]]
    params: Dict[str, Any] = {}


class WorkflowExecuteRequest(BaseModel):
    """Request for /workflow/execute endpoint - JSON workflow execution."""

    workflow_id: str
    inputs: Dict[str, Any]


class BaseUserRequestModel(BaseModel):
    """Base model for endpoints accepting natural language requests."""

    user_request: str = Field(
        ...,
        min_length=1,
        description="Natural language task description",
    )


class BaseEnrichModel(BaseModel):
    """Base model for enrichment endpoints."""

    data: Dict[str, Any] = Field(..., description="Record to enrich")


class BaseBulkModel(BaseModel):
    """Base model for bulk processing endpoints."""

    rows: List[Dict[str, Any]] = Field(..., description="Records to enrich (max 10,000)", min_length=1, max_length=10000)
    webhook_url: Optional[str] = Field(None, description="Webhook URL for completion")


class PlanRequest(BaseUserRequestModel):
    """Request for /plan endpoint - AI-powered execution planning."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_request": "Research Tesla and find contact information for their sales team"
                }
            ]
        }
    }


class OrchestrateRequest(BaseUserRequestModel):
    """Request for /orchestrate endpoint - full AI workflow orchestration."""

    stream: bool = Field(default=True, description="Enable SSE streaming")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_request": "Research Tesla and find contact information",
                    "stream": True,
                }
            ]
        }
    }


class WorkflowGenerateRequest(BaseUserRequestModel):
    """Request for /workflow/generate endpoint - AI workflow generation."""

    save_as: Optional[str] = Field(None, description="Workflow name to save")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_request": "Create a workflow to validate emails and check tech stack"
                }
            ]
        }
    }


class EnrichRequest(BaseEnrichModel):
    """Request for /enrich endpoint - multi-tool single record enrichment."""

    tools: List[str] = Field(..., description="Tool names to apply", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {"email": "john@example.com"},
                    "tools": ["email-intel"],
                }
            ]
        }
    }


class EnrichAutoRequest(BaseEnrichModel):
    """Request for /enrich/auto endpoint - auto-detect enrichment tools."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {
                        "email": "john@example.com",
                        "phone": "+14155552671",
                    }
                }
            ]
        }
    }


class BulkProcessRequest(BaseBulkModel):
    """Request for /bulk endpoint - bulk processing with specified tools."""

    tools: List[str] = Field(..., description="Tool names for all rows", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rows": [{"email": "john@example.com"}],
                    "tools": ["email-intel"],
                }
            ]
        }
    }


class BulkAutoProcessRequest(BaseBulkModel):
    """Request for /bulk/auto endpoint - bulk processing with auto-detection."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rows": [
                        {"email": "john@example.com", "phone": "+14155552671"}
                    ]
                }
            ]
        }
    }


class SavedJobCreateRequest(BaseModel):
    """Request for POST /jobs/saved endpoint - create saved job."""

    name: str = Field(..., min_length=1, description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    tool_name: str = Field(..., description="Tool to execute")
    params: Dict[str, Any] = Field(..., description="Tool parameters")
    is_template: bool = Field(default=False, description="Template job with variables")
    template_vars: Optional[List[str]] = Field(None, description="Template variable names")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Daily Phone Validation",
                    "tool_name": "phone-validation",
                    "params": {"phone_number": "+14155552671"},
                }
            ]
        }
    }


class JobScheduleUpdateRequest(BaseModel):
    """Request for PATCH /jobs/saved/{job_id}/schedule endpoint."""

    is_scheduled: bool = Field(..., description="Enable/disable scheduling")
    schedule_preset: Optional[str] = Field(
        None, description="Schedule preset: daily, weekly, monthly", pattern="^(daily|weekly|monthly)$"
    )
    schedule_cron: Optional[str] = Field(None, description="Custom cron expression")

    model_config = {
        "json_schema_extra": {
            "examples": [{"is_scheduled": True, "schedule_preset": "daily"}]
        }
    }
