"""Base Pydantic models for V2 API - DRY principle applied.

These base models are inherited by endpoint-specific models to avoid field duplication.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BaseUserRequestModel(BaseModel):
    """Base model for endpoints accepting natural language requests."""
    user_request: str = Field(
        ...,
        min_length=1,
        description="Natural language task description (e.g., 'Find email for john@company.com and validate it')"
    )


class BaseEnrichModel(BaseModel):
    """Base model for enrichment endpoints."""
    data: Dict[str, Any] = Field(
        ...,
        description="Record to enrich with any fields"
    )


class BaseBulkModel(BaseModel):
    """Base model for bulk processing endpoints."""
    rows: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of records to enrich (max 10,000 rows)"
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Optional webhook URL for completion notification"
    )
