"""Saved jobs routes for V2 API - tool-based saved jobs with scheduling."""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from v2.api.dependencies import get_tools_registry
from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware
from v2.api.models import JobScheduleUpdateRequest, SavedJobCreateRequest
from v2.infrastructure.auth import get_current_user
from v2.infrastructure.database.repositories.jobs import JobRepository
from v2.utils.scheduling import calculate_next_run_at
from v2.core.execution import ToolExecutor

router = APIRouter(tags=["Saved Jobs"])

# Initialize middleware
_quota = QuotaMiddleware()
_logging = APILoggingMiddleware()


def get_job_repository() -> JobRepository:
    """Dependency to get JobRepository instance."""
    return JobRepository()


@router.post("/jobs/saved")
async def create_saved_job(
    job_data: SavedJobCreateRequest,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
    job_repo: JobRepository = Depends(get_job_repository),
):
    """Create a new saved job.

    - **name**: Job name (required)
    - **tool_name**: Tool to execute (must exist in TOOLS registry)
    - **params**: Tool-specific parameters
    - **description**: Optional job description
    - **is_template**: If true, job can be used as template with variables
    - **template_vars**: List of template variable names
    """
    start_time = time.time()

    # Use anonymous user if not authenticated
    effective_user_id = user_id or "00000000-0000-0000-0000-000000000000"

    try:
        # Validate tool exists in registry
        if job_data.tool_name not in tools:
            error_response = {
                "success": False,
                "error": f"Tool '{job_data.tool_name}' not found in registry",
            }
            _logging.log_call(
                user_id=effective_user_id,
                tool_name="create_saved_job",
                tool_type="job_management",
                input_data=job_data.dict(),
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                error_message=f"Tool '{job_data.tool_name}' not found in registry",
            )
            return JSONResponse(status_code=400, content=error_response)

        # Create job via repository
        saved_job = job_repo.create_job(
            user_id=effective_user_id,
            name=job_data.name,
            tool_name=job_data.tool_name,
            params=job_data.params,
            description=job_data.description,
            is_template=job_data.is_template,
            template_vars=job_data.template_vars,
        )

        if not saved_job:
            error_response = {"success": False, "error": "Failed to create saved job"}
            _logging.log_call(
                user_id=effective_user_id,
                tool_name="create_saved_job",
                tool_type="job_management",
                input_data=job_data.dict(),
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                error_message="Failed to create saved job",
            )
            return JSONResponse(status_code=500, content=error_response)

        response = {
            "success": True,
            "data": {
                "id": saved_job.id,
                "name": saved_job.name,
                "tool_name": saved_job.tool_name,
                "description": saved_job.description,
                "is_template": saved_job.is_template,
                "created_at": saved_job.created_at,
            },
            "message": "Saved job created successfully",
        }

        _logging.log_call(
            user_id=effective_user_id,
            tool_name="create_saved_job",
            tool_type="job_management",
            input_data=job_data.dict(),
            output_data=response,
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {"success": False, "error": f"Failed to create job: {str(e)}"}
        _logging.log_call(
            user_id=effective_user_id,
            tool_name="create_saved_job",
            tool_type="job_management",
            input_data=job_data.dict(),
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)


@router.get("/jobs/saved")
async def list_saved_jobs(
    user_id: Optional[str] = Depends(get_current_user),
    job_repo: JobRepository = Depends(get_job_repository),
):
    """List all saved jobs for the current user.

    Returns jobs ordered by created_at (newest first).
    """
    start_time = time.time()

    # Use anonymous user if not authenticated
    effective_user_id = user_id or "00000000-0000-0000-0000-000000000000"

    try:
        jobs = job_repo.list_user_jobs(effective_user_id)

        response = {
            "success": True,
            "data": [
                {
                    "id": job.id,
                    "name": job.name,
                    "tool_name": job.tool_name,
                    "description": job.description,
                    "is_template": job.is_template,
                    "is_scheduled": job.is_scheduled,
                    "schedule_preset": job.schedule_preset,
                    "schedule_cron": job.schedule_cron,
                    "next_run_at": job.next_run_at,
                    "last_run_at": job.last_run_at,
                    "created_at": job.created_at,
                }
                for job in jobs
            ],
            "count": len(jobs),
        }

        _logging.log_call(
            user_id=effective_user_id,
            tool_name="list_saved_jobs",
            tool_type="job_management",
            input_data={},
            output_data={"count": len(jobs)},
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {"success": False, "error": f"Failed to list jobs: {str(e)}"}
        _logging.log_call(
            user_id=effective_user_id,
            tool_name="list_saved_jobs",
            tool_type="job_management",
            input_data={},
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)


@router.post("/jobs/saved/{job_id}/run")
async def run_saved_job(
    job_id: str,
    user_id: Optional[str] = Depends(get_current_user),
    tools: Dict[str, Any] = Depends(get_tools_registry),
    job_repo: JobRepository = Depends(get_job_repository),
):
    """Execute a saved job.

    - **job_id**: UUID of saved job to execute

    Quota: Counts as 1 API call toward monthly quota.
    """
    start_time = time.time()

    # Use anonymous user if not authenticated
    effective_user_id = user_id or "00000000-0000-0000-0000-000000000000"

    try:
        # Get job and verify ownership
        job = job_repo.get_job_by_id(job_id, effective_user_id)

        if not job:
            error_response = {
                "success": False,
                "error": f"Job {job_id} not found or access denied",
            }
            _logging.log_call(
                user_id=effective_user_id,
                tool_name="run_saved_job",
                tool_type="job_execution",
                input_data={"job_id": job_id},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                error_message=f"Job {job_id} not found or access denied",
            )
            return JSONResponse(status_code=404, content=error_response)

        # Quota enforcement for authenticated users
        if user_id:
            await _quota.check_quota(user_id)

        # Validate tool exists
        if job.tool_name not in tools:
            error_response = {
                "success": False,
                "error": f"Tool '{job.tool_name}' not found in registry",
            }
            _logging.log_call(
                user_id=effective_user_id,
                tool_name="run_saved_job",
                tool_type="job_execution",
                input_data={"job_id": job_id, "tool_name": job.tool_name},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                error_message=f"Tool '{job.tool_name}' not found in registry",
            )
            return JSONResponse(status_code=400, content=error_response)

        # Execute tool using ToolExecutor
        executor = ToolExecutor(tools)
        tool_result = await executor.execute(job.tool_name, job.params)

        # Update last_run_at timestamp
        job_repo.update_last_run_at(job_id)

        response = {
            "success": tool_result.get("success", False),
            "job_id": job_id,
            "job_name": job.name,
            "tool_name": job.tool_name,
            "result": tool_result.get("data"),
            "error": tool_result.get("error"),
            "executed_at": datetime.now().isoformat() + "Z",
        }

        _logging.log_call(
            user_id=effective_user_id,
            tool_name="run_saved_job",
            tool_type="job_execution",
            input_data={"job_id": job_id, "tool_name": job.tool_name, "params": job.params},
            output_data=response,
            success=tool_result.get("success", False),
            processing_ms=int((time.time() - start_time) * 1000),
            error_message=tool_result.get("error"),
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {"success": False, "error": f"Failed to run job: {str(e)}"}
        _logging.log_call(
            user_id=effective_user_id,
            tool_name="run_saved_job",
            tool_type="job_execution",
            input_data={"job_id": job_id},
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)


@router.patch("/jobs/saved/{job_id}/schedule")
async def update_job_schedule(
    job_id: str,
    schedule_data: JobScheduleUpdateRequest,
    user_id: Optional[str] = Depends(get_current_user),
    job_repo: JobRepository = Depends(get_job_repository),
):
    """Update job scheduling settings.

    - **job_id**: UUID of saved job to update
    - **is_scheduled**: Enable or disable scheduling
    - **schedule_preset**: Schedule preset (daily, weekly, monthly) - required if is_scheduled=true
    - **schedule_cron**: Custom cron expression (alternative to preset)
    """
    start_time = time.time()

    # Use anonymous user if not authenticated
    effective_user_id = user_id or "00000000-0000-0000-0000-000000000000"

    try:
        # Get job and verify ownership
        job = job_repo.get_job_by_id(job_id, effective_user_id)

        if not job:
            error_response = {
                "success": False,
                "error": f"Job {job_id} not found or access denied",
            }
            _logging.log_call(
                user_id=effective_user_id,
                tool_name="update_job_schedule",
                tool_type="job_management",
                input_data={"job_id": job_id, **schedule_data.dict()},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                error_message=f"Job {job_id} not found or access denied",
            )
            return JSONResponse(status_code=404, content=error_response)

        # Validate scheduling parameters
        if schedule_data.is_scheduled:
            if not schedule_data.schedule_preset and not schedule_data.schedule_cron:
                error_response = {
                    "success": False,
                    "error": "schedule_preset or schedule_cron required when is_scheduled=true",
                }
                _logging.log_call(
                    user_id=effective_user_id,
                    tool_name="update_job_schedule",
                    tool_type="job_management",
                    input_data={"job_id": job_id, **schedule_data.dict()},
                    output_data=error_response,
                    success=False,
                    processing_ms=int((time.time() - start_time) * 1000),
                    error_message="schedule_preset or schedule_cron required when is_scheduled=true",
                )
                return JSONResponse(status_code=400, content=error_response)

        # Calculate next_run_at if scheduling is enabled
        next_run_at = None
        if schedule_data.is_scheduled:
            schedule_expr = schedule_data.schedule_preset or schedule_data.schedule_cron
            if schedule_expr:  # Ensure schedule_expr is not None
                try:
                    next_run = calculate_next_run_at(schedule_expr)
                    next_run_at = next_run.isoformat() if next_run else None
                except ValueError as e:
                    error_response = {"success": False, "error": f"Invalid schedule: {str(e)}"}
                    _logging.log_call(
                        user_id=effective_user_id,
                        tool_name="update_job_schedule",
                        tool_type="job_management",
                        input_data={"job_id": job_id, **schedule_data.dict()},
                        output_data=error_response,
                        success=False,
                        processing_ms=int((time.time() - start_time) * 1000),
                        error_message=str(e),
                    )
                    return JSONResponse(status_code=400, content=error_response)

        # Update schedule via repository
        updated_job = job_repo.update_schedule(
            job_id=job_id,
            is_scheduled=schedule_data.is_scheduled,
            schedule_preset=schedule_data.schedule_preset,
            schedule_cron=schedule_data.schedule_cron,
            next_run_at=next_run_at,
        )

        if not updated_job:
            error_response = {"success": False, "error": "Failed to update job schedule"}
            _logging.log_call(
                user_id=effective_user_id,
                tool_name="update_job_schedule",
                tool_type="job_management",
                input_data={"job_id": job_id, **schedule_data.dict()},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                error_message="Failed to update job schedule",
            )
            return JSONResponse(status_code=500, content=error_response)

        response = {
            "success": True,
            "data": {
                "id": updated_job.id,
                "is_scheduled": updated_job.is_scheduled,
                "schedule_preset": updated_job.schedule_preset,
                "schedule_cron": updated_job.schedule_cron,
                "next_run_at": updated_job.next_run_at,
                "updated_at": updated_job.updated_at,
            },
            "message": "Job schedule updated successfully",
        }

        _logging.log_call(
            user_id=effective_user_id,
            tool_name="update_job_schedule",
            tool_type="job_management",
            input_data={"job_id": job_id, **schedule_data.dict()},
            output_data=response,
            success=True,
            processing_ms=int((time.time() - start_time) * 1000),
        )

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {
            "success": False,
            "error": f"Failed to update schedule: {str(e)}",
        }
        _logging.log_call(
            user_id=effective_user_id,
            tool_name="update_job_schedule",
            tool_type="job_management",
            input_data={"job_id": job_id, **schedule_data.dict()},
            output_data=error_response,
            success=False,
            processing_ms=int((time.time() - start_time) * 1000),
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content=error_response)
