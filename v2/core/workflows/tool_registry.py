"""Tool registry for V2 API.

Extensible tool registry supporting internal functions, HTTP webhooks, and MCP tools.
"""

import os
import time
from typing import Any, Dict, Optional

from v2.core.execution.tool_executor import ToolExecutor
from v2.core.logging import logger


class ToolRegistry:
    """Extensible tool registry supporting internal functions, HTTP webhooks, and MCP tools.

    Loads tool definitions from database with caching for performance.
    Follows Repository pattern: Abstracts data access for tool definitions.
    """

    def __init__(self, internal_tools: Dict[str, Dict[str, Any]]):
        """Initialize ToolRegistry with internal TOOLS registry.

        Args:
            internal_tools: Existing TOOLS dict (from V1 Modal functions)
        """
        self.internal_tools = internal_tools
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

    async def execute(
        self, tool_name: str, params: Dict[str, Any], user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute tool by name (internal, HTTP, or MCP).

        Args:
            tool_name: Tool name from tool_definitions table
            params: Tool parameters
            user_id: User ID for HTTP tool authentication

        Returns:
            Dict: {success, tool_name, tool_type, data/error, execution_time_ms}
        """
        start_time = time.time()

        # Load tool definition from DB (with caching)
        tool_def = await self._load_tool_definition(tool_name)

        if not tool_def:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' not found in registry",
                "error_type": "KeyError",
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        tool_type = tool_def["tool_type"]

        try:
            if tool_type == "internal":
                result = await self._execute_internal(tool_def, params)
            elif tool_type == "http":
                result = await self._execute_http(tool_def, params, user_id)
            elif tool_type == "mcp":
                return {
                    "success": False,
                    "tool_name": tool_name,
                    "tool_type": tool_type,
                    "error": "MCP tools not implemented yet (V2 feature)",
                    "error_type": "NotImplementedError",
                    "execution_time_ms": (time.time() - start_time) * 1000,
                }
            else:
                return {
                    "success": False,
                    "tool_name": tool_name,
                    "tool_type": tool_type,
                    "error": f"Unknown tool type: {tool_type}",
                    "error_type": "ValueError",
                    "execution_time_ms": (time.time() - start_time) * 1000,
                }

            execution_time = (time.time() - start_time) * 1000

            return {
                "success": result.get("success", True),
                "tool_name": tool_name,
                "tool_type": tool_type,
                "data": result.get("data"),
                "error": result.get("error"),
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": execution_time,
            }

    async def _load_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Load tool definition from database with 5-minute caching.

        Args:
            tool_name: Tool name to load

        Returns:
            Tool definition dict or None if not found
        """
        from supabase import create_client

        # Check cache
        if tool_name in self._cache:
            age = time.time() - self._cache_timestamp.get(tool_name, 0)
            if age < self._cache_ttl:
                return self._cache[tool_name]

        # Load from database
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            # Supabase not configured - tool registry disabled
            logger.warning("supabase_not_configured", action="load_tool_definition")
            return None

        try:
            supabase = create_client(supabase_url, supabase_key)
            response = (
                supabase.table("tool_definitions")
                .select("*")
                .eq("tool_name", tool_name)
                .eq("is_active", True)
                .execute()
            )

            if response.data and len(response.data) > 0:
                tool_def = response.data[0]
                # Cache it
                self._cache[tool_name] = tool_def
                self._cache_timestamp[tool_name] = time.time()
                return tool_def

            return None

        except Exception as e:
            logger.warning("tool_definition_load_failed", tool_name=tool_name, error=str(e))
            return None

    async def _execute_internal(self, tool_def: Dict, params: Dict) -> Dict[str, Any]:
        """Execute internal tool via existing TOOLS registry.

        Args:
            tool_def: Tool definition from database
            params: Tool parameters

        Returns:
            Dict: {success, data/error}
        """
        tool_name = tool_def["tool_name"]

        # Check if tool exists in internal registry
        if tool_name not in self.internal_tools:
            return {
                "success": False,
                "error": f"Internal tool '{tool_name}' not found in TOOLS registry",
            }

        # Use existing ToolExecutor
        executor = ToolExecutor(self.internal_tools)
        result = await executor.execute(tool_name, params)

        return {"success": result["success"], "data": result.get("data"), "error": result.get("error")}

    async def _execute_http(
        self, tool_def: Dict, params: Dict, user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute HTTP webhook/API call with user authentication.

        Args:
            tool_def: Tool definition with HTTP config
            params: Tool parameters
            user_id: User ID for integration lookup

        Returns:
            Dict: {success, data/error}
        """
        import httpx
        from supabase import create_client

        config = tool_def["config"]
        http_method = config.get("http_method", "POST")
        http_url = config.get("http_url")
        http_url_template = config.get("http_url_template")
        requires_integration = config.get("requires_integration")

        # Resolve URL template if needed
        if http_url_template:
            # Load user integration
            if not user_id:
                return {"success": False, "error": "HTTP tool requires authentication (user_id)"}

            # Get integration config
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                return {"success": False, "error": "Supabase not configured for HTTP tools"}

            try:
                supabase = create_client(supabase_url, supabase_key)
                response = (
                    supabase.table("user_integrations")
                    .select("*")
                    .eq("user_id", user_id)
                    .eq("integration_name", requires_integration)
                    .eq("is_active", True)
                    .execute()
                )

                if not response.data or len(response.data) == 0:
                    return {
                        "success": False,
                        "error": f"Integration '{requires_integration}' not configured for user",
                    }

                integration_config = response.data[0]["config"]

                # Simple template substitution ({{user_integrations.X.Y}})
                http_url = http_url_template.replace(
                    f"{{{{user_integrations.{requires_integration}.webhook_url}}}}",
                    integration_config.get("webhook_url", ""),
                )

            except Exception as e:
                return {"success": False, "error": f"Failed to load integration: {str(e)}"}

        # Execute HTTP request
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if http_method.upper() == "POST":
                    response = await client.post(http_url, json=params)
                elif http_method.upper() == "GET":
                    response = await client.get(http_url, params=params)
                else:
                    return {"success": False, "error": f"Unsupported HTTP method: {http_method}"}

                response.raise_for_status()

                return {
                    "success": True,
                    "data": (
                        response.json()
                        if response.headers.get("content-type", "").startswith("application/json")
                        else {"response": response.text}
                    ),
                }

        except Exception as e:
            return {"success": False, "error": f"HTTP request failed: {str(e)}"}
