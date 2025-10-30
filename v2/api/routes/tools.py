"""Dynamic tool routes for V2 API.

Registers tool routes using RouteFactory pattern.
"""

from typing import Any, Dict

from fastapi import FastAPI

from v2.api.factory import RouteFactory


def register_tool_routes(app: FastAPI, tools_registry: Dict[str, Any]) -> None:
    """Register dynamic tool routes for all tools in registry.

    Creates routes at /{tool_type}/{tool_name} for each tool.

    Args:
        app: FastAPI application instance
        tools_registry: Dictionary of tool configurations with:
            - fn: Tool function
            - type: Tool type (enrichment, generation, analysis)
            - params: Parameter specifications
            - doc: Documentation string
            - tag: OpenAPI tag

    Example:
        >>> register_tool_routes(app, TOOLS)
        # Creates routes like:
        # POST /enrichment/email-intel
        # POST /generation/blog-create
        # POST /analysis/aeo-health-check
    """
    factory = RouteFactory()

    for tool_name, tool_config in tools_registry.items():
        app.add_api_route(
            f"/{tool_config['type']}/{tool_name}",
            factory.create_tool_route(tool_name, tool_config),
            methods=["POST"],
            tags=[tool_config["tag"]],
            summary=tool_config["doc"].split("\n\n")[0],
        )
