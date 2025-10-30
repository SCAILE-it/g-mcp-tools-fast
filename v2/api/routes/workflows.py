"""Workflow routes for V2 API - JSON workflow execution and generation."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware, check_rate_limit
from v2.api.models import WorkflowExecuteRequest, WorkflowGenerateRequest
from v2.api.utils import create_sse_event
from v2.core.logging import logger
from v2.core.workflows import ToolRegistry, WorkflowExecutor
from v2.infrastructure.auth import get_current_user

router = APIRouter(tags=["Workflows"])

# Initialize middleware
_quota = QuotaMiddleware()
_logging = APILoggingMiddleware()


def _get_system_context(request: Request) -> Dict[str, Any]:
    """Extract system context from request headers."""
    return {
        "date": datetime.now().isoformat()[:10],
        "datetime": datetime.now().isoformat(),
        "country": request.headers.get("cf-ipcountry", "US"),
        "timezone": request.headers.get("timezone", "UTC"),
        "language": (
            request.headers.get("accept-language", "en")[:2]
            if request.headers.get("accept-language")
            else "en"
        ),
    }


@router.post("/workflow/execute")
async def workflow_execute_route(
    request: Request,
    request_data: WorkflowExecuteRequest,
    user_id: Optional[str] = Depends(get_current_user),
):
    """Execute JSON-based workflow with SSE streaming.

    - **workflow_id**: Workflow template UUID from workflow_templates table
    - **inputs**: User-provided input values matching workflow schema

    SSE Events:
    - **step_start**: Step execution started
    - **step_complete**: Step execution completed (with result or error)
    - **complete**: All steps finished
    """
    # Rate limiting check
    if not await check_rate_limit(user_id, "/workflow/execute", 10):
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Rate limit exceeded. Maximum 10 requests per minute.",
                "retry_after": 60,
            },
        )

    # Import TOOLS registry
    TOOLS: Dict[str, Any] = {}  # TODO: Get from app.state or dependency

    try:
        # Get system context from request headers
        system_context = _get_system_context(request)

        # Initialize workflow executor
        tool_registry = ToolRegistry(TOOLS)
        workflow_executor = WorkflowExecutor(tool_registry)

        # SSE streaming
        async def event_generator():
            final_result = None
            start_time = time.time()

            try:
                async for event in workflow_executor.execute(
                    workflow_id=request_data.workflow_id,
                    inputs=request_data.inputs,
                    system_context=system_context,
                    user_id=user_id,
                ):
                    event_type = event.get("event", "message")
                    event_data = event.get("data", {})

                    # Track final result for logging
                    if event_type == "complete":
                        final_result = event_data

                    # Create SSE event
                    yield create_sse_event(event_type, event_data)

            except Exception as e:
                yield create_sse_event("error", {"error": str(e)})
                final_result = {"error": str(e), "successful": 0, "failed": 1}

            finally:
                # Log to api_calls table
                if user_id and final_result:
                    processing_ms = int((time.time() - start_time) * 1000)

                    _logging.log_call(
                        user_id=user_id,
                        tool_name=f"workflow:{request_data.workflow_id}",
                        tool_type="workflow",
                        input_data={"inputs": request_data.inputs},
                        output_data=final_result,
                        success=final_result.get("failed", 0) == 0,
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


@router.post("/workflow/generate")
async def workflow_generate_route(
    request_data: WorkflowGenerateRequest,
    user_id: Optional[str] = Depends(get_current_user),
):
    """Generate workflow JSON from natural language using AI."""
    try:
        user_request = request_data.user_request
        save_as = request_data.save_as

        # Load system documentation from database
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Supabase not configured"},
            )

        supabase = create_client(supabase_url, supabase_key)

        # Fetch documentation
        docs_response = supabase.table("system_documentation").select("*").execute()

        docs_content = (
            "\n\n".join(
                [f"# {doc['title']}\n\n{doc['content']}" for doc in docs_response.data]
            )
            if docs_response.data
            else ""
        )

        # Fetch available tools
        tools_response = (
            supabase.table("tool_definitions")
            .select("tool_name, display_name, description, tool_type")
            .eq("is_active", True)
            .execute()
        )

        tools_list = (
            "\n".join(
                [
                    f"- **{tool['tool_name']}** ({tool['tool_type']}): {tool['description']}"
                    for tool in tools_response.data
                ]
            )
            if tools_response.data
            else ""
        )

        # Build system prompt for Gemini
        system_prompt = f"""You are a workflow generator AI. Generate valid JSON workflows from natural language descriptions.

# System Documentation

{docs_content}

# Available Tools

{tools_list}

# Task

Generate a valid JSON workflow for this request:
{user_request}

Return ONLY the JSON workflow object, no markdown code blocks or explanations."""

        # Call Gemini
        from google import generativeai as genai

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Gemini API key not configured"},
            )

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        response = model.generate_content(system_prompt)
        workflow_json_str = response.text.strip()

        # Remove markdown code blocks if present
        if workflow_json_str.startswith("```"):
            workflow_json_str = workflow_json_str.split("```")[1]
            if workflow_json_str.startswith("json"):
                workflow_json_str = workflow_json_str[4:]
            workflow_json_str = workflow_json_str.strip()

        # Parse and validate JSON
        try:
            workflow_json = json.loads(workflow_json_str)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Generated invalid JSON: {str(e)}",
                "raw_output": workflow_json_str,
            }

        # Validate schema structure
        if "steps" not in workflow_json:
            return {
                "success": False,
                "error": "Generated workflow missing 'steps' array",
            }

        # Save if requested
        saved_id = None
        if save_as and user_id:
            try:
                save_response = (
                    supabase.table("workflow_templates")
                    .insert(
                        {
                            "name": save_as,
                            "description": f"AI-generated from: {user_request[:100]}",
                            "json_schema": workflow_json,
                            "scope": "global",
                            "is_system": False,
                            "created_by": user_id,
                        }
                    )
                    .execute()
                )

                if save_response.data and len(save_response.data) > 0:
                    saved_id = save_response.data[0]["id"]

            except Exception as e:
                # Saving failed but generation succeeded
                logger.warning("workflow_save_failed", error=str(e))

        return {
            "success": True,
            "workflow": workflow_json,
            "saved_id": saved_id,
            "metadata": {
                "user_request": user_request,
                "generated_at": datetime.now().isoformat() + "Z",
            },
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.get("/tools")
async def tools_list_route(user_id: Optional[str] = Depends(get_current_user)):
    """List all available tools from tool_definitions table."""
    try:
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Supabase not configured"},
            )

        supabase = create_client(supabase_url, supabase_key)

        # Fetch all active tools
        response = (
            supabase.table("tool_definitions")
            .select("tool_name, tool_type, category, display_name, description")
            .eq("is_active", True)
            .execute()
        )

        if not response.data:
            return {"success": True, "tools": [], "total": 0}

        # Group by category
        by_category: Dict[str, list] = {}
        for tool in response.data:
            category = tool["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(
                {
                    "name": tool["tool_name"],
                    "type": tool["tool_type"],
                    "display_name": tool["display_name"],
                    "description": tool["description"],
                }
            )

        return {
            "success": True,
            "tools": response.data,
            "by_category": by_category,
            "total": len(response.data),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.get("/workflow/documentation")
async def workflow_documentation_route():
    """Get system documentation for workflow JSON schema."""
    try:
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Supabase not configured"},
            )

        supabase = create_client(supabase_url, supabase_key)

        # Fetch all documentation
        response = supabase.table("system_documentation").select("*").execute()

        if not response.data:
            return {"success": True, "documentation": []}

        return {"success": True, "documentation": response.data}

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )
