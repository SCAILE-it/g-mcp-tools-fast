"""Error classifier for V2 API.

Classifies errors into categories for intelligent retry decisions.
"""

import asyncio

from v2.core.retry_config import ErrorCategory


class ErrorClassifier:
    """Classifies errors into categories for intelligent retry decisions.

    Follows Single Responsibility Principle: Only handles error categorization.
    Static methods for stateless classification.
    """

    @staticmethod
    def categorize(error: Exception) -> ErrorCategory:
        """Categorize an error into TRANSIENT, RATE_LIMIT, PERMANENT, or UNKNOWN.

        Args:
            error: Exception to categorize

        Returns:
            ErrorCategory enum value
        """
        import requests

        error_str = str(error).lower()

        # Check for transient network errors
        if isinstance(
            error,
            (
                asyncio.TimeoutError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ),
        ):
            return ErrorCategory.TRANSIENT

        # Check for HTTP errors in error message
        if "http error" in error_str or "status code" in error_str:
            # Rate limiting
            if "429" in error_str or "rate limit" in error_str:
                return ErrorCategory.RATE_LIMIT

            # Transient server errors
            if any(code in error_str for code in ["503", "504", "502"]):
                return ErrorCategory.TRANSIENT

            # Permanent client errors
            if any(code in error_str for code in ["400", "401", "403", "404", "405"]):
                return ErrorCategory.PERMANENT

        # Check for permanent validation errors
        if isinstance(error, ValueError):
            return ErrorCategory.PERMANENT

        # Check for timeout in message
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorCategory.TRANSIENT

        # Default to unknown (no retry)
        return ErrorCategory.UNKNOWN

    @staticmethod
    def categorize_by_type(error_type: str, error_msg: str) -> ErrorCategory:
        """Categorize error by type name and message (for ToolExecutor results).

        Args:
            error_type: Exception type name (e.g., "TimeoutError", "ValueError")
            error_msg: Error message string

        Returns:
            ErrorCategory enum value
        """
        error_msg_lower = error_msg.lower()

        # Transient network errors
        if error_type in ["TimeoutError", "Timeout", "ConnectionError"]:
            return ErrorCategory.TRANSIENT

        # Permanent validation errors
        if error_type == "ValueError":
            return ErrorCategory.PERMANENT

        # Check HTTP status codes in message
        if "http error" in error_msg_lower or "status code" in error_msg_lower:
            # Rate limiting
            if "429" in error_msg or "rate limit" in error_msg_lower:
                return ErrorCategory.RATE_LIMIT

            # Transient server errors
            if any(code in error_msg for code in ["503", "504", "502"]):
                return ErrorCategory.TRANSIENT

            # Permanent client errors
            if any(code in error_msg for code in ["400", "401", "403", "404", "405"]):
                return ErrorCategory.PERMANENT

        # Check timeout in message
        if "timeout" in error_msg_lower or "timed out" in error_msg_lower:
            return ErrorCategory.TRANSIENT

        # Default to unknown
        return ErrorCategory.UNKNOWN
