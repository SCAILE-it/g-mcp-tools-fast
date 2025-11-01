"""Tool executor for V2 API.

Executes tools from registry with parameter validation and error handling.
"""

import asyncio
import time
from typing import Any, Dict, List


class ToolExecutor:
    """Executes tools from TOOLS registry.

    Handles parameter validation, error handling, and execution tracking.
    Follows Single Responsibility Principle: Only handles tool execution.
    """

    def __init__(self, tools: Dict[str, Dict[str, Any]]):
        """Initialize ToolExecutor with tools registry.

        Args:
            tools: TOOLS registry dict {tool_name: {fn, type, params, ...}}
        """
        self.tools = tools

    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given parameters.

        Args:
            tool_name: Name of tool to execute
            params: Parameters dict for the tool

        Returns:
            Dict with: success, tool_name, tool_type, tool_tag, data/error, execution_time_ms
        """
        start_time = time.time()

        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' not found in registry",
                "error_type": "KeyError",
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        tool_config = self.tools[tool_name]
        tool_fn = tool_config["fn"]
        tool_type = tool_config.get("type", "unknown")
        tool_tag = tool_config.get("tag", "")
        param_specs = tool_config.get("params", [])

        # Validate and prepare parameters
        try:
            kwargs = self._prepare_params(param_specs, params)
        except ValueError as e:
            return {
                "success": False,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_tag": tool_tag,
                "error": str(e),
                "error_type": "ValueError",
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        # Execute tool
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**kwargs)
            else:
                # Run sync function in thread pool
                result = await asyncio.to_thread(tool_fn, **kwargs)

            execution_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_tag": tool_tag,
                "data": result,
                "execution_time_ms": execution_time,
            }
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_tag": tool_tag,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": execution_time,
            }

    def _prepare_params(
        self, param_specs: List[tuple], provided_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and prepare parameters based on param specs.

        Args:
            param_specs: List of (name, type, required, default?) tuples
            provided_params: User-provided parameters

        Returns:
            Dict of validated parameters

        Raises:
            ValueError: If required param missing or validation fails
        """
        kwargs = {}

        for spec in param_specs:
            param_name = spec[0]
            # param_type = spec[1]  # Unused in V1 implementation
            is_required = spec[2]
            has_default = len(spec) > 3
            default_value = spec[3] if has_default else None

            if param_name in provided_params:
                # Use provided value
                kwargs[param_name] = provided_params[param_name]
            elif is_required:
                # Required param missing
                raise ValueError(f"Missing required parameter: '{param_name}'")
            elif has_default:
                # Use default value
                kwargs[param_name] = default_value
            # else: optional param not provided, don't include

        return kwargs
