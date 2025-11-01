"""Orchestration routes for V2 API - AI-powered planning and execution."""

import time
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from v2.api.dependencies import get_tools_registry
from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware, check_rate_limit
from v2.api.models import ExecuteRequest, OrchestrateRequest, PlanRequest
from v2.api.utils import create_sse_event
from v2.core.execution import ToolExecutor
from v2.core.orchestration import Orchestrator, Planner
from v2.infrastructure.auth import get_current_user

router = APIRouter(tags=["AI Orchestration"])

# Initialize middleware
_quota = QuotaMiddleware()
_logging = APILoggingMiddleware()


async def _process_rows_with_progress(
    tool_name: str, rows: List[Dict[str, Any]], params: Dict[str, Any], tools_registry: Dict[str, Any]
) -> AsyncIterator[Dict[str, Any]]:
    """Process rows with single tool, yielding progress events.

    Yields SSE events:
        - {"event": "progress", "data": {"processed": N, "total": M, "percentage": P}}
        - {"event": "result", "data": {"results": [...], "summary": {...}}}
    """
    total = len(rows)
    results = []
    executor = ToolExecutor(tools_registry)

    # Process rows in chunks for progress updates
    CHUNK_SIZE = 10
    for i in range(0, total, CHUNK_SIZE):
        chunk = rows[i : i + CHUNK_SIZE]

        # Process chunk
        for row_idx, row in enumerate(chunk):
            # Build params for this row (merge row data + provided params)
            row_params = {**params, **row}

            # Execute tool
            result = await executor.execute(tool_name, row_params)
            results.append(
                {
                    "row_index": i + row_idx,
                    "success": result["success"],
                    "data": result.get("data"),
                    "error": result.get("error"),
                }
            )

        # Emit progress
        processed = min(i + CHUNK_SIZE, total)
        percentage = int((processed / total) * 100)
        yield {
            "event": "progress",
            "data": {"processed": processed, "total": total, "percentage": percentage},
        }

    # Emit final result
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    yield {
        "event": "result",
        "data": {
            "results": results,
            "summary": {"total": total, "successful": successful, "failed": failed},
            "success": failed == 0,
        },
    }


@router.post("/plan")
async def plan_route(
    request_data: PlanRequest, user_id: Optional[str] = Depends(get_current_user)
):
    """Generate execution plan from user request using AI planning.

    Returns:
        - **success**: bool
        - **plan**: List of step descriptions
        - **total_steps**: Number of steps in plan
        - **sources**: List of data sources used (files, APIs, web) - NEW
    """
    # Rate limiting check
    if not await check_rate_limit(user_id, "/plan", 10):
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Rate limit exceeded. Maximum 10 requests per minute.",
                "retry_after": 60,
            },
        )

    start_time = time.time()
    try:
        # Import file processing utilities (lazy import to avoid circular dependencies)
        from v2.utils.file_processor import FileProcessor
        from v2.utils.sources_tracker import SourcesTracker

        user_request = request_data.user_request
        company_context = request_data.company_context
        enabled_tools = request_data.enabled_tools
        files = request_data.files or []
        request_user_id = request_data.user_id or user_id

        # Process uploaded files if provided
        processed_files = []
        if files:
            processed_files = await FileProcessor.process_files(files)

        # Initialize sources tracker
        sources_tracker = SourcesTracker.from_files(processed_files)

        # Enhance user_request with company context if provided
        enhanced_request = user_request
        if company_context:
            context_str = "\n".join(
                [f"{key}: {value}" for key, value in company_context.items()]
            )
            enhanced_request = f"{user_request}\n\nCompany Context:\n{context_str}"

        # Initialize Planner
        planner = Planner()

        # Generate plan with enabled_tools filtering and file context
        plan = planner.generate(
            enhanced_request,
            enabled_tools=enabled_tools,
            file_context=processed_files,
        )

        # Add reasoning for complex plans (2+ steps)
        reasoning = None
        if len(plan) >= 2:
            reasoning = (
                f"This is a multi-step workflow that requires {len(plan)} steps. "
                f"Each step will be executed sequentially to complete your request."
            )

        # Log successful API call
        processing_ms = int((time.time() - start_time) * 1000)
        _logging.log_call(
            user_id=request_user_id or "00000000-0000-0000-0000-000000000000",
            tool_name="/plan",
            tool_type="orchestration",
            input_data={
                "user_request": user_request,
                "has_company_context": company_context is not None,
                "enabled_tools": enabled_tools,
                "files_count": len(files),
            },
            output_data={"plan": plan, "total_steps": len(plan)},
            success=True,
            processing_ms=processing_ms,
        )

        response = {
            "success": True,
            "plan": {"steps": plan},
            "total_steps": len(plan),
            "sources": sources_tracker.get_sources(),  # NEW: Track data sources
            "metadata": {
                "user_request": user_request,
                "generated_at": datetime.now().isoformat() + "Z",
            },
        }

        # Add reasoning only for complex plans
        if reasoning:
            response["reasoning"] = reasoning

        return response
    except Exception as e:
        # Log failed API call
        processing_ms = int((time.time() - start_time) * 1000)
        _logging.log_call(
            user_id=user_id or "00000000-0000-0000-0000-000000000000",
            tool_name="/plan",
            tool_type="orchestration",
            input_data=request_data.dict(),
            output_data={},
            success=False,
            error_message=str(e),
            processing_ms=processing_ms,
        )

        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/execute")
async def execute_route(
    request_data: ExecuteRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
):
    """Execute single tool on multiple rows with SSE streaming.

    Request:
    - **executionId**: Unique execution identifier
    - **tool**: Tool name (from TOOLS registry)
    - **data**: Array of rows to process
    - **params**: Tool-specific parameters

    SSE Events:
    - **progress**: Row-level progress updates
    - **result**: Final results with summary
    """
    # Rate limiting check
    if not await check_rate_limit(user_id, "/execute", 20):
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Rate limit exceeded. Maximum 20 requests per minute.",
                "retry_after": 60,
            },
        )

    try:
        # Quota enforcement for authenticated users
        if user_id:
            await _quota.check_quota(user_id)

        # Validate tool exists
        if request_data.tool not in tools:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Tool '{request_data.tool}' not found in registry",
                },
            )

        # SSE streaming
        async def event_generator():
            final_result = None
            start_time = time.time()

            try:
                async for event in _process_rows_with_progress(
                    request_data.tool, request_data.data, request_data.params, tools
                ):
                    event_type = event.get("event", "message")
                    event_data = event.get("data", {})

                    # Track final result for logging
                    if event_type == "result":
                        final_result = event_data

                    # Create SSE event
                    yield create_sse_event(event_type, event_data)

            except Exception as e:
                # Create error event
                yield create_sse_event("error", {"error": str(e)})
                final_result = {
                    "error": str(e),
                    "successful": 0,
                    "failed": len(request_data.data),
                }

            finally:
                # Log usage for authenticated users after streaming completes
                if user_id and final_result:
                    processing_ms = int((time.time() - start_time) * 1000)
                    tool_config = tools.get(request_data.tool, {})
                    tool_type = tool_config.get("type", "unknown")

                    _logging.log_call(
                        user_id=user_id,
                        tool_name=request_data.tool,
                        tool_type=tool_type,
                        input_data={
                            "executionId": request_data.executionId,
                            "total_rows": len(request_data.data),
                            "params": request_data.params,
                        },
                        output_data=final_result,
                        success=final_result.get("success", False),
                        processing_ms=processing_ms,
                        error_message=final_result.get("error"),
                    )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/orchestrate")
async def orchestrate_route(
    request_data: OrchestrateRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
):
    """Orchestrate full AI workflow with SSE streaming.

    SSE Events:
    - **plan_init**: Initial plan with steps
    - **step_start**: Step execution started
    - **step_complete**: Step execution completed (with result or error)
    - **complete**: All steps finished

    Returns:
        StreamingResponse with SSE events (if stream=true)
        OR blocking JSON response (if stream=false)
    """
    # Rate limiting check
    if not await check_rate_limit(user_id, "/orchestrate", 10):
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Rate limit exceeded. Maximum 10 requests per minute.",
                "retry_after": 60,
            },
        )

    try:
        user_request = request_data.user_request
        stream_mode = request_data.stream

        # Extract plan_steps from request_data.plan if provided
        plan_steps = None
        if request_data.plan:
            plan_steps = request_data.plan.get("steps")

        # Get company_context and user_id from request
        company_context = request_data.company_context
        request_user_id = request_data.user_id or user_id

        # Quota enforcement for authenticated users
        if request_user_id:
            await _quota.check_quota(request_user_id)

        # Initialize Orchestrator with TOOLS registry
        orchestrator = Orchestrator(tools)

        # Streaming mode (SSE)
        if stream_mode:

            async def event_generator():
                final_result = None
                try:
                    async for event in orchestrator.execute_plan_stream(
                        user_request,
                        plan_steps=plan_steps,
                        company_context=company_context
                    ):
                        event_type = event.get("event", "message")
                        event_data = event.get("data", {})

                        # Track final result for logging
                        if event_type == "complete":
                            final_result = event_data

                        # Create SSE event
                        yield create_sse_event(event_type, event_data)

                except Exception as e:
                    # Create error event
                    yield create_sse_event("error", {"error": str(e)})
                    final_result = {"error": str(e), "successful": 0, "failed": 0}

                finally:
                    # Log usage for authenticated users after streaming completes
                    if request_user_id and final_result:
                        _logging.log_call(
                            user_id=request_user_id,
                            tool_name="orchestrator",
                            tool_type="ai_orchestration",
                            input_data={
                                "user_request": user_request,
                                "stream": True,
                                "has_plan": plan_steps is not None,
                                "has_company_context": company_context is not None,
                            },
                            output_data=final_result,
                            success=final_result.get("successful", 0) > 0,
                            processing_ms=0,
                            error_message=final_result.get("error"),
                        )

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Blocking mode (regular JSON response)
        else:
            result = await orchestrator.execute_plan(
                user_request,
                plan_steps=plan_steps,
                company_context=company_context
            )

            # Log usage for authenticated users
            if request_user_id:
                _logging.log_call(
                    user_id=request_user_id,
                    tool_name="orchestrator",
                    tool_type="ai_orchestration",
                    input_data={
                        "user_request": user_request,
                        "has_plan": plan_steps is not None,
                        "has_company_context": company_context is not None,
                    },
                    output_data=result,
                    success=result["success"],
                    processing_ms=0,
                    error_message=None,
                )

            return result

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )
