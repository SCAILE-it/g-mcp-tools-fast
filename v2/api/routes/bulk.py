"""Bulk processing routes for V2 API - parallel enrichment with batch jobs."""

import secrets
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from v2.api.dependencies import get_tools_registry
from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware
from v2.api.models import BulkAutoProcessRequest, BulkProcessRequest
from v2.core.batch import BatchProcessor
from v2.infrastructure.auth import get_current_user

router = APIRouter(tags=["Bulk Processing"])

# Initialize middleware
_quota = QuotaMiddleware()
_logging = APILoggingMiddleware()

# In-memory batch results storage
# TODO: Move to database or Redis for production
_batch_results: Dict[str, Dict[str, Any]] = {}


async def _process_batch_internal(
    batch_id: str,
    rows: list,
    auto_mode: bool,
    tool_names: Optional[list],
    webhook_url: Optional[str],
    tools_registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Internal batch processing with BatchProcessor.

    Args:
        batch_id: Unique batch identifier
        rows: List of records to process
        auto_mode: If True, auto-detect tools; if False, use tool_names
        tool_names: List of tool names (for explicit mode)
        webhook_url: Optional webhook for completion notification
        tools_registry: TOOLS registry

    Returns:
        Dict with batch processing results
    """
    # Initialize BatchProcessor
    batch_processor = BatchProcessor()

    # Process batch
    result_dict = await batch_processor.process(
        rows=rows,
        auto_detect=auto_mode,
        tool_names=tool_names,
        tool_map=tools_registry,
        webhook_url=webhook_url,
        batch_id=batch_id,
    )

    # Store results
    _batch_results[batch_id] = {
        "status": "completed",
        "total_rows": len(rows),
        "successful": result_dict.get("successful", 0),
        "failed": result_dict.get("failed", 0),
        "processing_time_seconds": round(result_dict.get("processing_time_seconds", 0.0), 2),
        "processing_mode": result_dict.get("processing_mode", "unknown"),
        "results": result_dict.get("results", []),
        "completed_at": datetime.now().isoformat() + "Z",
    }

    # Send webhook if provided
    if webhook_url:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json={
                        "batch_id": batch_id,
                        "status": "completed",
                        "summary": {
                            "total": len(rows),
                            "successful": result_dict.get("successful", 0),
                            "failed": result_dict.get("failed", 0),
                        },
                    },
                    timeout=5.0,
                )
        except Exception:
            # Webhook failure doesn't fail the batch
            pass

    return _batch_results[batch_id]


@router.post("/bulk")
async def bulk_process(
    request_data: BulkProcessRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
):
    """Process multiple records in parallel with specified tools.

    Quota: Each row counts as 1 API call toward monthly quota.

    - **rows**: List of records to enrich (max 10,000)
    - **tools**: List of tool names to apply to ALL rows
    - **webhook_url**: Optional webhook URL for completion notification
    """
    start_time = time.time()

    rows = request_data.rows
    tool_names = request_data.tools
    webhook_url = request_data.webhook_url

    # Quota enforcement: Bulk = N API calls (one per row)
    if user_id:
        for _ in range(len(rows)):
            await _quota.check_quota(user_id)

    try:
        batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"
        result = await _process_batch_internal(
            batch_id, rows, False, tool_names, webhook_url, tools
        )

        response = {
            "success": True,
            "batch_id": batch_id,
            "status": result.get("status", "completed"),
            "total_rows": result.get("total_rows", len(rows)),
            "successful": result.get("successful", 0),
            "failed": result.get("failed", 0),
            "processing_time_seconds": result.get("processing_time_seconds", 0),
            "processing_mode": result.get("processing_mode", "unknown"),
            "results": result.get("results", []),
            "message": "Batch processing completed successfully.",
            "metadata": {"timestamp": datetime.now().isoformat() + "Z"},
        }

        _logging.log_call(
            tool_name="bulk",
            tool_type="enrichment",
            input_data={"rows_count": len(rows), "tools": tool_names},
            output_data={
                "batch_id": batch_id,
                "successful": response["successful"],
                "failed": response["failed"],
            },
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {"success": False, "error": f"Batch processing failed: {str(e)}"}
        _logging.log_call(
            tool_name="bulk",
            tool_type="enrichment",
            input_data={"rows_count": len(rows), "tools": tool_names},
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)


@router.post("/bulk/auto")
async def bulk_auto_process(
    request_data: BulkAutoProcessRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
):
    """Process multiple records in parallel with auto-detection.

    Quota: Each row counts as 1 API call toward monthly quota.

    - **rows**: List of records to enrich (max 10,000)
    - **webhook_url**: Optional webhook URL for completion notification
    """
    start_time = time.time()

    rows = request_data.rows
    webhook_url = request_data.webhook_url

    # Quota enforcement: Bulk = N API calls (one per row)
    if user_id:
        for _ in range(len(rows)):
            await _quota.check_quota(user_id)

    try:
        batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"
        result = await _process_batch_internal(batch_id, rows, True, None, webhook_url, tools)

        response = {
            "success": True,
            "batch_id": batch_id,
            "status": result.get("status", "completed"),
            "total_rows": result.get("total_rows", len(rows)),
            "successful": result.get("successful", 0),
            "failed": result.get("failed", 0),
            "processing_time_seconds": result.get("processing_time_seconds", 0),
            "processing_mode": result.get("processing_mode", "unknown"),
            "results": result.get("results", []),
            "message": "Batch auto-processing completed successfully.",
            "metadata": {"timestamp": datetime.now().isoformat() + "Z"},
        }

        _logging.log_call(
            tool_name="bulk-auto",
            tool_type="enrichment",
            input_data={"rows_count": len(rows)},
            output_data={
                "batch_id": batch_id,
                "successful": response["successful"],
                "failed": response["failed"],
            },
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {
            "success": False,
            "error": f"Batch auto-processing failed: {str(e)}",
        }
        _logging.log_call(
            tool_name="bulk-auto",
            tool_type="enrichment",
            input_data={"rows_count": len(rows)},
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            user_id=user_id,
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)


@router.get("/bulk/status/{batch_id}")
async def bulk_status(batch_id: str, user_id: Optional[str] = Depends(get_current_user)):
    """Check the status of a batch processing job.

    - **batch_id**: Batch ID returned from /bulk or /bulk/auto
    """
    try:
        if batch_id not in _batch_results:
            return JSONResponse(
                status_code=404, content={"success": False, "error": "Batch not found"}
            )

        batch_data = _batch_results[batch_id]
        return JSONResponse(
            content={
                "success": True,
                "batch_id": batch_id,
                "status": batch_data.get("status", "unknown"),
                "total_rows": batch_data.get("total_rows", 0),
                "successful": batch_data.get("successful", 0),
                "failed": batch_data.get("failed", 0),
                "processing_time_seconds": batch_data.get("processing_time_seconds", 0),
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"},
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Status check failed: {str(e)}"},
        )


@router.get("/bulk/results/{batch_id}")
async def bulk_results(batch_id: str, user_id: Optional[str] = Depends(get_current_user)):
    """Download results from a completed batch job.

    - **batch_id**: Batch ID returned from /bulk or /bulk/auto
    """
    try:
        if batch_id not in _batch_results:
            return JSONResponse(
                status_code=404, content={"success": False, "error": "Batch not found"}
            )

        batch_data = _batch_results[batch_id]

        if batch_data.get("status") != "completed":
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Batch not completed yet. Status: {batch_data.get('status', 'unknown')}",
                },
            )

        return JSONResponse(
            content={
                "success": True,
                "batch_id": batch_id,
                "status": "completed",
                "results": batch_data.get("results", []),
                "total_rows": batch_data.get("total_rows", 0),
                "metadata": {
                    "completed_at": batch_data.get("completed_at"),
                    "timestamp": datetime.now().isoformat() + "Z",
                },
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Results retrieval failed: {str(e)}"},
        )
