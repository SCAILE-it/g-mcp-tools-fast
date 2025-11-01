"""Execution tracker for V2 API.

Handles workflow execution persistence in database.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

from v2.core.logging import logger


class ExecutionTracker:
    """Tracks workflow execution state in database.

    Follows Single Responsibility Principle: Only handles DB persistence.
    Separates data access from business logic for testability.
    """

    async def load_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow template from database.

        Args:
            workflow_id: Workflow template ID

        Returns:
            Workflow template dict, or None if not found
        """
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("supabase_not_configured", action="load_workflow")
            return None

        try:
            supabase = create_client(supabase_url, supabase_key)
            response = (
                supabase.table("workflow_templates").select("*").eq("id", workflow_id).execute()
            )

            if response.data and len(response.data) > 0:
                return response.data[0]

            return None

        except Exception as e:
            logger.warning("workflow_load_failed", workflow_id=workflow_id, error=str(e))
            return None

    async def create_execution(
        self, template_id: str, inputs: Dict[str, Any], user_id: Optional[str]
    ) -> str:
        """Create workflow_executions record in database.

        Args:
            template_id: Workflow template ID
            inputs: User-provided input values
            user_id: Optional user ID

        Returns:
            Execution ID (or "unknown" if DB unavailable)
        """
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("supabase_not_configured", action="create_execution")
            return "unknown"

        try:
            supabase = create_client(supabase_url, supabase_key)
            response = (
                supabase.table("workflow_executions")
                .insert(
                    {
                        "template_id": template_id,
                        "inputs": inputs,
                        "status": "running",
                        "started_at": datetime.now().isoformat(),
                    }
                )
                .execute()
            )

            if response.data and len(response.data) > 0:
                return response.data[0]["id"]

            return "unknown"

        except Exception as e:
            logger.warning(
                "execution_record_create_failed", template_id=template_id, error=str(e)
            )
            return "unknown"

    async def complete_execution(
        self, execution_id: str, outputs: Dict[str, Any], status: str, processing_ms: int
    ) -> None:
        """Update workflow_executions record with completion data.

        Args:
            execution_id: Execution record ID
            outputs: Workflow output values
            status: Final status ("completed", "failed", etc.)
            processing_ms: Processing time in milliseconds
        """
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        # Skip if DB not configured or execution_id is unknown
        if not supabase_url or not supabase_key or execution_id == "unknown":
            return

        try:
            supabase = create_client(supabase_url, supabase_key)
            supabase.table("workflow_executions").update(
                {
                    "outputs": outputs,
                    "status": status,
                    "completed_at": datetime.now().isoformat(),
                    "processing_time_ms": processing_ms,
                }
            ).eq("id", execution_id).execute()

        except Exception as e:
            logger.warning("execution_record_update_failed", execution_id=execution_id, error=str(e))
