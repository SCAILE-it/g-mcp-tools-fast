"""Route factory for V2 API.

Factory pattern for creating FastAPI route handlers with consistent middleware.
"""

import time
from typing import Any, Callable, Dict, Optional

from fastapi import Body, Depends
from fastapi.responses import JSONResponse

from v2.api.middleware import APILoggingMiddleware, QuotaMiddleware
from v2.infrastructure.auth import get_current_user


class RouteFactory:
    """Factory for creating tool route handlers.

    Applies consistent middleware (quota, logging) to all routes.
    Follows Factory pattern for clean separation of concerns.
    """

    def __init__(
        self,
        quota_middleware: Optional[QuotaMiddleware] = None,
        logging_middleware: Optional[APILoggingMiddleware] = None,
    ):
        """Initialize route factory.

        Args:
            quota_middleware: QuotaMiddleware instance (optional)
            logging_middleware: APILoggingMiddleware instance (optional)
        """
        self.quota = quota_middleware or QuotaMiddleware()
        self.logging = logging_middleware or APILoggingMiddleware()

    def create_tool_route(self, tool_name: str, tool_config: Dict[str, Any]) -> Callable:
        """Create a FastAPI route handler for a tool.

        Args:
            tool_name: Name of the tool (e.g., "email-intel")
            tool_config: Tool configuration dictionary with:
                - fn: Tool function to execute
                - type: Tool type (enrichment, generation, analysis)
                - params: List of parameter specs [(name, type, required, default?)]
                - doc: Tool documentation string

        Returns:
            Async route handler function

        Example:
            >>> factory = RouteFactory()
            >>> route = factory.create_tool_route("email-intel", {
            ...     "fn": email_intel,
            ...     "type": "enrichment",
            ...     "params": [("email", str, True)],
            ...     "doc": "Check email registrations"
            ... })
        """

        async def handler(
            request_data: Dict[str, Any] = Body(...),
            user_id: Optional[str] = Depends(get_current_user),
        ):
            start_time = time.time()

            # Quota enforcement (middleware)
            await self.quota.check_quota(user_id)

            # Parse and validate parameters
            kwargs = self._parse_parameters(request_data, tool_config["params"])

            # Check for validation errors
            if isinstance(kwargs, dict) and "error" in kwargs:
                return JSONResponse(status_code=400, content={"success": False, **kwargs})

            # Execute tool with error handling
            try:
                result = await tool_config["fn"](**kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = {"success": False, "error": str(e)}
                success = False
                error_msg = str(e)

            # Calculate processing time
            processing_ms = int((time.time() - start_time) * 1000)

            # Log API call (middleware)
            tokens_used = (
                result.get("metadata", {}).get("total_tokens", 0) if success else 0
            )
            self.logging.log_call(
                user_id=user_id,
                tool_name=tool_name,
                tool_type=tool_config["type"],
                input_data=request_data,
                output_data=result,
                success=success,
                processing_ms=processing_ms,
                error_message=error_msg,
                tokens_used=tokens_used,
            )

            # Return appropriate status code
            if not success:
                return JSONResponse(status_code=500, content=result)

            return JSONResponse(content=result)

        # Set handler metadata
        handler.__doc__ = tool_config["doc"]
        handler.__name__ = f"{tool_name.replace('-', '_')}_route"
        return handler

    def _parse_parameters(
        self, request_data: Dict[str, Any], param_specs: list
    ) -> Dict[str, Any]:
        """Parse and validate request parameters.

        Args:
            request_data: Request body data
            param_specs: List of parameter specifications

        Returns:
            Dictionary of parsed kwargs, or error dict if validation fails
        """
        kwargs = {}

        for param_config in param_specs:
            param_name = param_config[0]
            is_required = param_config[2]
            value = request_data.get(param_name)

            # Check required parameters
            if is_required and not value:
                return {"error": f"{param_name} required"}

            # Apply default value if parameter has one
            if len(param_config) == 4:
                # Has default value
                kwargs[param_name] = value if value is not None else param_config[3]
            elif value is not None:
                # No default, but value provided
                kwargs[param_name] = value

        return kwargs
