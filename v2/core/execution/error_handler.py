"""Error handler for V2 API.

Intelligent error handler with retry and fallback strategies.
"""

import asyncio
from typing import Any, Dict, List

from v2.core.execution.error_classifier import ErrorClassifier
from v2.core.execution.tool_executor import ToolExecutor
from v2.core.retry_config import RetryConfig


class ErrorHandler:
    """Intelligent error handler with retry and fallback strategies.

    Follows SOLID principles:
    - Single Responsibility: Handles error recovery only
    - Open/Closed: Extensible via RetryConfig
    - Liskov Substitution: Can replace ToolExecutor in any context
    - Interface Segregation: Separate methods for retry vs fallback
    - Dependency Inversion: Depends on ToolExecutor abstraction

    Uses composition pattern to wrap ToolExecutor without modifying it.
    """

    def __init__(self, executor: ToolExecutor, config: RetryConfig):
        """Initialize ErrorHandler with executor and retry configuration.

        Args:
            executor: ToolExecutor instance to wrap
            config: RetryConfig with retry behavior settings
        """
        self.executor = executor
        self.config = config

    async def execute_with_retry(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with automatic retry on transient failures.

        Uses exponential backoff: delay = initial_delay * (backoff_factor ^ retry_count)
        Caps delay at max_delay to prevent excessive waits.

        Args:
            tool_name: Name of tool to execute
            params: Parameters dict for the tool

        Returns:
            Dict with: success, tool_name, data/error, retry_count, execution_time_ms
        """
        retry_count = 0

        for attempt in range(self.config.max_retries + 1):
            # Execute tool
            result = await self.executor.execute(tool_name, params)

            # Success - return immediately
            if result["success"]:
                result["retry_count"] = retry_count
                return result

            # Failure - check if we should retry
            if attempt >= self.config.max_retries:
                break

            # Categorize error using type and message from ToolExecutor
            error_type = result.get("error_type", "Exception")
            error_msg = result.get("error", "Unknown error")
            error_category = ErrorClassifier.categorize_by_type(error_type, error_msg)

            # Don't retry permanent or unknown errors
            if error_category not in self.config.retry_on:
                result["retry_count"] = retry_count
                return result

            # Calculate exponential backoff delay
            delay = min(
                self.config.initial_delay * (self.config.backoff_factor**retry_count),
                self.config.max_delay,
            )

            # Wait before retry
            await asyncio.sleep(delay)
            retry_count += 1

        # All retries exhausted - return last error
        result["retry_count"] = retry_count
        return result

    async def execute_with_fallback(
        self, primary_tool: str, fallback_tools: List[str], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool with fallback to alternative tools on failure.

        Tries primary tool first. If it fails, tries each fallback tool in order
        until one succeeds or all fail.

        Args:
            primary_tool: Primary tool name to try first
            fallback_tools: List of fallback tool names to try on failure
            params: Parameters dict for all tools

        Returns:
            Dict with: success, tool_name, data/error, used_fallback, fallback_tool
        """
        # Try primary tool first
        result = await self.executor.execute(primary_tool, params)

        if result["success"]:
            result["used_fallback"] = False
            return result

        # Primary failed - try fallbacks
        for fallback_tool in fallback_tools:
            result = await self.executor.execute(fallback_tool, params)

            if result["success"]:
                result["used_fallback"] = True
                result["fallback_tool"] = fallback_tool
                return result

        # All fallbacks failed - return last error
        result["used_fallback"] = False
        return result
