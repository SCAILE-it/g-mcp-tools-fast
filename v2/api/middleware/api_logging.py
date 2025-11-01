"""API call logging middleware for V2 API.

Provides reusable API call logging for FastAPI routes.
"""

from typing import Any, Dict, Optional

from v2.infrastructure.database.models import APICallRecord
from v2.infrastructure.database.repositories import APICallRepository


class APILoggingMiddleware:
    """Middleware for logging API calls to database."""

    def __init__(self, api_call_repository: Optional[APICallRepository] = None):
        """Initialize API logging middleware.

        Args:
            api_call_repository: APICallRepository instance (optional)
        """
        self.repo = api_call_repository or APICallRepository()

    def log_call(
        self,
        user_id: Optional[str],
        tool_name: str,
        tool_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool,
        processing_ms: int,
        error_message: Optional[str] = None,
        tokens_used: int = 0,
    ) -> None:
        """Log an API call.

        Args:
            user_id: User UUID (None for anonymous)
            tool_name: Name of the tool called
            tool_type: Type of tool (enrichment, generation, analysis)
            input_data: Request input data
            output_data: Response output data
            success: Whether call succeeded
            processing_ms: Processing time in milliseconds
            error_message: Error message if failed
            tokens_used: Number of AI tokens used
        """
        record = APICallRecord(
            user_id=user_id,
            tool_name=tool_name,
            tool_type=tool_type,
            input_data=input_data,
            output_data=output_data,
            success=success,
            processing_ms=processing_ms,
            error_message=error_message,
            tokens_used=tokens_used,
        )

        self.repo.log_call(record)
