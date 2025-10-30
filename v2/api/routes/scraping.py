"""Scraping routes for V2 API."""

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware
from v2.api.models import ScrapeRequest
from v2.api.utils import get_cache, set_cache
from v2.infrastructure.auth import get_current_user
from v2.tools.scraping.flexible_scraper import FlexibleScraper, FlexibleScraperError

router = APIRouter(tags=["Web Scraping"])

# Initialize middleware
_quota = QuotaMiddleware()
_logging = APILoggingMiddleware()


@router.post("/scrape")
async def scrape_route(
    request_data: Dict[str, Any],
    user_id: Optional[str] = Depends(get_current_user),
):
    """Extract structured data from websites using AI.

    - **url**: Website URL to scrape
    - **prompt**: Natural language extraction instruction
    - **schema**: Optional JSON schema for structured output
    - **max_pages**: Number of pages to scrape (1-50)
    - **auto_discover_pages**: Auto-discover relevant pages
    """
    start_time = time.time()

    # Quota enforcement
    if user_id:
        await _quota.check_quota(user_id)

    # Validate request
    try:
        scrape_request = ScrapeRequest(**request_data)
    except Exception as e:
        error_response = {
            "success": False,
            "error": f"Invalid request: {str(e)}",
            "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"},
        }

        # Log failed request
        processing_ms = int((time.time() - start_time) * 1000)
        _logging.log_call(
            tool_name="scrape",
            tool_type="enrichment",
            input_data=request_data,
            output_data=error_response,
            success=False,
            processing_ms=processing_ms,
            user_id=user_id,
            error_message=str(e),
        )

        return JSONResponse(status_code=400, content=error_response)

    # Check cache
    cached_result = get_cache(
        scrape_request.url, scrape_request.prompt, scrape_request.output_schema
    )
    if cached_result:
        processing_ms = int((time.time() - start_time) * 1000)
        response = {
            "success": True,
            "data": cached_result["data"],
            "metadata": {
                **cached_result["metadata"],
                "cached": True,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Log cached response
        _logging.log_call(
            tool_name="scrape",
            tool_type="enrichment",
            input_data=request_data,
            output_data=response,
            success=True,
            processing_ms=processing_ms,
            user_id=user_id,
        )

        return JSONResponse(content=response)

    # Get Gemini API key
    api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get(
        "GEMINI_API_KEY"
    )
    if not api_key:
        error_response = {"success": False, "error": "Missing Gemini API key"}
        processing_ms = int((time.time() - start_time) * 1000)

        _logging.log_call(
            tool_name="scrape",
            tool_type="enrichment",
            input_data=request_data,
            output_data=error_response,
            success=False,
            processing_ms=processing_ms,
            user_id=user_id,
            error_message="Missing Gemini API key",
        )

        return JSONResponse(status_code=500, content=error_response)

    # Execute scraping
    try:
        scraper = FlexibleScraper(api_key=api_key)

        actions_list = None
        if scrape_request.actions:
            actions_list = [action.dict() for action in scrape_request.actions]

        extracted_data = await scraper.scrape(
            url=scrape_request.url,
            prompt=scrape_request.prompt,
            schema=scrape_request.output_schema,
            actions=actions_list,
            max_pages=scrape_request.max_pages or 1,
            timeout=scrape_request.timeout or 30,
            extract_links=scrape_request.extract_links or False,
            use_context_analysis=scrape_request.use_context_analysis if scrape_request.use_context_analysis is not None else True,
            auto_discover_pages=scrape_request.auto_discover_pages or False,
        )

        extraction_time = time.time() - start_time
        pages_scraped = (
            extracted_data.pop("_pages_scraped", 1)
            if isinstance(extracted_data, dict)
            else 1
        )

        # Cache result
        cache_value = {
            "data": extracted_data,
            "metadata": {
                "extraction_time": extraction_time,
                "pages_scraped": pages_scraped,
            },
            "timestamp": datetime.now(),
        }
        set_cache(
            scrape_request.url,
            scrape_request.prompt,
            scrape_request.output_schema,
            cache_value,
        )

        response = {
            "success": True,
            "data": extracted_data,
            "metadata": {
                "extraction_time": round(extraction_time, 2),
                "pages_scraped": pages_scraped,
                "cached": False,
                "model": FlexibleScraper.DEFAULT_MODEL,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Log successful response
        processing_ms = int((time.time() - start_time) * 1000)
        _logging.log_call(
            tool_name="scrape",
            tool_type="enrichment",
            input_data=request_data,
            output_data=response,
            success=True,
            processing_ms=processing_ms,
            user_id=user_id,
        )

        return JSONResponse(content=response)

    except FlexibleScraperError as e:
        error_response = {
            "success": False,
            "error": str(e),
            "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"},
        }

        processing_ms = int((time.time() - start_time) * 1000)
        _logging.log_call(
            tool_name="scrape",
            tool_type="enrichment",
            input_data=request_data,
            output_data=error_response,
            success=False,
            processing_ms=processing_ms,
            user_id=user_id,
            error_message=str(e),
        )

        return JSONResponse(status_code=400, content=error_response)

    except Exception as e:
        error_response = {
            "success": False,
            "error": "An unexpected error occurred. Please try again or contact support.",
            "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"},
        }

        processing_ms = int((time.time() - start_time) * 1000)
        _logging.log_call(
            tool_name="scrape",
            tool_type="enrichment",
            input_data=request_data,
            output_data=error_response,
            success=False,
            processing_ms=processing_ms,
            user_id=user_id,
            error_message=str(e),
        )

        return JSONResponse(status_code=500, content=error_response)
