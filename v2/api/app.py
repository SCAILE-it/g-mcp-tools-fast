"""FastAPI application factory for V2 API."""

from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from v2.api.middleware import request_id_middleware
from v2.api.routes import bulk, enrichment, jobs, orchestration, scraping, system, workflows
from v2.api.routes.tools import register_tool_routes


def create_app(tools_registry: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI app. Factory pattern for testability."""
    app = FastAPI(
        title="g-mcp-tools-fast",
        description=(
            "GTM power API with 13 tools organized by category: enrichment (8), "
            "generation (3), and analysis (2). Web scraping, email intel, company data, "
            "phone validation, tech stack detection, AI research, content creation, "
            "and SEO/AEO analysis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add middleware
    app.middleware("http")(request_id_middleware)

    # Register routes
    app.include_router(system.router)
    app.include_router(orchestration.router)
    app.include_router(workflows.router)
    app.include_router(enrichment.router)
    app.include_router(bulk.router)
    app.include_router(scraping.router)
    app.include_router(jobs.router)

    # Register dynamic tool routes (if tools registry provided)
    if tools_registry:
        register_tool_routes(app, tools_registry)

    # Store tools registry for route access
    if tools_registry:
        app.state.tools_registry = tools_registry

    # Custom OpenAPI schema with logo
    def custom_openapi():
        """Generate custom OpenAPI schema. Cached after first call."""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="g-mcp-tools-fast API",
            version="1.0.0",
            description="GTM power API: enrichment, generation, and analysis tools",
            routes=app.routes,
        )

        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    setattr(app, "openapi", custom_openapi)

    return app
