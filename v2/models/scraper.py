"""Scraper-related Pydantic models for V2 API.

These models handle web scraping requests with AI-powered extraction.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """Browser action types for FlexibleScraper."""
    CLICK = "click"
    SCROLL = "scroll"
    WAIT = "wait"
    TYPE = "type"
    SCREENSHOT = "screenshot"


class ScrapeAction(BaseModel):
    """A single browser action to perform during scraping."""
    type: ActionType
    selector: Optional[str] = None
    text: Optional[str] = None
    milliseconds: Optional[int] = None
    pixels: Optional[int] = None


class ScrapeRequest(BaseModel):
    """Request for /scrape endpoint - AI-powered web scraping with Gemini extraction."""
    url: str
    prompt: str = Field(..., min_length=1)
    output_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")  # Renamed to avoid BaseModel conflict
    actions: Optional[List[ScrapeAction]] = None
    max_pages: Optional[int] = Field(1, ge=1, le=50)
    timeout: Optional[int] = Field(30, ge=5, le=120)
    extract_links: Optional[bool] = False
    use_context_analysis: Optional[bool] = True
    auto_discover_pages: Optional[bool] = False

    class Config:
        populate_by_name = True  # Allow both "schema" and "output_schema"

    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Ensure URL has http:// or https:// prefix."""
        if not v.startswith(("http://", "https://")):
            v = f"https://{v}"
        return v.strip()
