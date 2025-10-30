"""Workflow-related Pydantic models for V2 API.

These models handle workflow execution and generation requests.
"""

from typing import Any, Dict, List
from pydantic import BaseModel


class ExecuteRequest(BaseModel):
    """Request for /execute endpoint - single-tool batch processing with SSE streaming."""
    executionId: str
    tool: str
    data: List[Dict[str, Any]]
    params: Dict[str, Any] = {}


class WorkflowExecuteRequest(BaseModel):
    """Request for /workflow/execute endpoint - JSON workflow execution with SSE streaming."""
    workflow_id: str
    inputs: Dict[str, Any]
