"""Enrichment-related Pydantic models for V2 API.

These models handle single and bulk enrichment requests.
"""

from typing import List
from pydantic import Field
from v2.models.base import BaseEnrichModel, BaseBulkModel


class EnrichRequest(BaseEnrichModel):
    """Request for /enrich endpoint - multi-tool single record enrichment."""
    tools: List[str] = Field(
        ...,
        min_items=1,
        description="List of tool names to apply (e.g., ['email-intel', 'phone-validation'])"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {"email": "john@example.com", "company": "Acme Corp"},
                    "tools": ["email-intel", "company-data"]
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
                        "domain": "example.com"
                    }
                }
            ]
        }
    }


class BulkProcessRequest(BaseBulkModel):
    """Request for /bulk endpoint - bulk processing with specified tools."""
    tools: List[str] = Field(..., min_items=1, description="List of tool names to apply to ALL rows")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rows": [{"email": "john@example.com"}, {"email": "jane@example.com"}],
                    "tools": ["email-intel", "email-validate"]
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
                        {"email": "john@example.com", "phone": "+14155552671"},
                        {"domain": "example.com"}
                    ]
                }
            ]
        }
    }
