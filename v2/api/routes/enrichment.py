"""Enrichment routes for V2 API - single record enrichment with auto-detection."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from v2.api.dependencies import get_tools_registry
from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware
from v2.api.models import EnrichAutoRequest, EnrichRequest
from v2.infrastructure.auth import get_current_user

router = APIRouter(tags=["Enrichment"])

# Initialize middleware
_quota = QuotaMiddleware()
_logging = APILoggingMiddleware()


def _detect_field_type(key: str, value: Any) -> str:
    """Detect field type from key/value for auto-enrichment."""
    key_lower = key.lower()

    # Email detection
    if "email" in key_lower and isinstance(value, str) and "@" in value:
        return "email"

    # Phone detection
    if "phone" in key_lower and isinstance(value, str):
        return "phone"

    # Domain detection
    if ("domain" in key_lower or "website" in key_lower) and isinstance(value, str):
        if value.startswith(("http://", "https://")):

            return "domain"  # Will need to extract domain from URL
        return "domain"

    # Company detection
    if ("company" in key_lower or "organization" in key_lower) and isinstance(value, str):
        return "company"

    # GitHub user detection
    if "github" in key_lower and isinstance(value, str):
        return "github_user"

    return "unknown"


def _auto_detect_enrichments(data: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    """Auto-detect which enrichment tools to apply.

    Returns:
        List of (tool_name, field_name, field_value) tuples
    """
    tool_specs = []

    for key, value in data.items():
        field_type = _detect_field_type(key, value)

        # Map field types to tools
        if field_type == "email":
            tool_specs.append(("email-intel", key, value))
        elif field_type == "phone":
            tool_specs.append(("phone-validation", key, value))
        elif field_type == "domain":
            tool_specs.extend(
                [
                    ("email-pattern", key, value),
                    ("whois", key, value),
                    ("tech-stack", key, value),
                ]
            )
        elif field_type == "company":
            tool_specs.append(("company-data", key, value))
        elif field_type == "github_user":
            tool_specs.append(("github-intel", key, value))

    return tool_specs


async def _run_enrichments(
    data: Dict[str, Any], tool_specs: List[Tuple[str, str, Any]], tools_registry: Dict[str, Any]
) -> Dict[str, Any]:
    """Run enrichment tools and merge results into data.

    Args:
        data: Original record
        tool_specs: List of (tool_name, field_name, field_value) tuples
        tools_registry: TOOLS registry

    Returns:
        Enriched data dictionary
    """
    result = data.copy()

    for tool_name, field_name, field_value in tool_specs:
        if tool_name not in tools_registry:
            continue

        tool_config = tools_registry[tool_name]
        tool_fn = tool_config["fn"]

        try:
            # Execute tool with field value
            enrichment_result = await tool_fn(**{field_name: field_value})

            # Merge enrichment data
            if enrichment_result.get("success") and "data" in enrichment_result:
                # Add enrichment data with tool prefix
                result[f"{tool_name}_data"] = enrichment_result["data"]

        except Exception:
            # Skip failed enrichments
            continue

    return result


@router.post("/enrich")
async def multi_tool_enrich(
    request_data: EnrichRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
):
    """Enrich a single record with multiple tools."""
    start_time = time.time()

    # Quota enforcement for authenticated users
    if user_id:
        await _quota.check_quota(user_id)

    data = request_data.data
    tool_names = request_data.tools

    if not isinstance(data, dict):
        error_response = {"success": False, "error": "data required (must be dict)"}
        _logging.log_call(
            tool_name="enrich",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message="data required (must be dict)",
        )
        return JSONResponse(status_code=400, content=error_response)

    if not tool_names or not isinstance(tool_names, list):
        error_response = {"success": False, "error": "tools required (must be list)"}
        _logging.log_call(
            tool_name="enrich",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message="tools required (must be list)",
        )
        return JSONResponse(status_code=400, content=error_response)

    try:
        # Build tool specs from explicit tool names
        tool_specs = []
        for tool_name in tool_names:
            for key, value in data.items():
                field_type = _detect_field_type(key, value)
                if (
                    (tool_name == "phone-validation" and field_type == "phone")
                    or (tool_name == "email-intel" and field_type == "email")
                    or (tool_name == "email-pattern" and field_type == "domain")
                    or (tool_name == "whois" and field_type == "domain")
                    or (tool_name == "tech-stack" and field_type == "domain")
                    or (tool_name == "company-data" and field_type == "company")
                    or (tool_name == "github-intel" and field_type == "github_user")
                ):
                    tool_specs.append((tool_name, key, value))
                    break

        if not tool_specs:
            error_response = {
                "success": False,
                "error": "No matching fields found for specified tools",
            }
            _logging.log_call(
                tool_name="enrich",
                tool_type="enrichment",
                input_data=request_data.dict(),
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="No matching fields found",
            )
            return JSONResponse(status_code=400, content=error_response)

        result = await _run_enrichments(data, tool_specs, tools)
        response = {
            "success": True,
            "data": result,
            "metadata": {
                "source": "multi-tool-enrich",
                "timestamp": datetime.now().isoformat() + "Z",
            },
        }

        _logging.log_call(
            tool_name="enrich",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=response,
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {"success": False, "error": f"Enrichment failed: {str(e)}"}
        _logging.log_call(
            tool_name="enrich",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)


@router.post("/enrich/auto")
async def auto_enrich(
    request_data: EnrichAutoRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
):
    """Auto-detect and enrich a single record with appropriate tools."""
    start_time = time.time()

    # Quota enforcement for authenticated users
    if user_id:
        await _quota.check_quota(user_id)

    data = request_data.data
    if not isinstance(data, dict):
        error_response = {"success": False, "error": "data required (must be dict)"}
        _logging.log_call(
            tool_name="enrich-auto",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message="data required (must be dict)",
        )
        return JSONResponse(status_code=400, content=error_response)

    try:
        tool_specs = _auto_detect_enrichments(data)
        if not tool_specs:
            response = {
                "success": True,
                "data": data,
                "metadata": {
                    "source": "auto-enrich",
                    "message": "No enrichments detected",
                    "timestamp": datetime.now().isoformat() + "Z",
                },
            }
            _logging.log_call(
                tool_name="enrich-auto",
                tool_type="enrichment",
                input_data=request_data.dict(),
                output_data=response,
                success=True,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
            )
            return JSONResponse(content=response)

        result = await _run_enrichments(data, tool_specs, tools)
        response = {
            "success": True,
            "data": result,
            "metadata": {
                "source": "auto-enrich",
                "tools_applied": len(tool_specs),
                "timestamp": datetime.now().isoformat() + "Z",
            },
        }

        _logging.log_call(
            tool_name="enrich-auto",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=response,
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {"success": False, "error": f"Auto-enrichment failed: {str(e)}"}
        _logging.log_call(
            tool_name="enrich-auto",
            tool_type="enrichment",
            input_data=request_data.dict(),
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)
