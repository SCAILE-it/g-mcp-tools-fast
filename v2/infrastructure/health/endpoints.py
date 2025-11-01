"""Health check endpoint handler for V2 API.

Provides /health endpoint with dependency testing.
"""

import asyncio
from typing import Any, Dict, List

from v2.infrastructure.health.checks import test_gemini_connection, test_supabase_connection


async def get_health_status(
    tool_count: int, tool_categories: List[str], service_name: str = "g-mcp-tools-fast-v2"
) -> Dict[str, Any]:
    """Get comprehensive health status.

    Args:
        tool_count: Total number of tools available
        tool_categories: List of tool categories (enrichment, generation, analysis)
        service_name: Service name for response

    Returns:
        Dict with overall status and dependency health
    """
    # Test dependencies in parallel
    gemini_health, supabase_health = await asyncio.gather(
        test_gemini_connection(), test_supabase_connection(), return_exceptions=True
    )

    # Handle exceptions from gather
    if isinstance(gemini_health, Exception):
        gemini_health = {"status": "error", "error": str(gemini_health)}
    if isinstance(supabase_health, Exception):
        supabase_health = {"status": "error", "error": str(supabase_health)}

    # Determine overall status
    all_healthy = (
        gemini_health.get("status") == "healthy"
        and supabase_health.get("status") == "healthy"
    )
    overall_status = "healthy" if all_healthy else "degraded"

    return {
        "status": overall_status,
        "service": service_name,
        "version": "2.0.0",
        "tools": tool_count,
        "categories": tool_categories,
        "dependencies": {"gemini": gemini_health, "supabase": supabase_health},
    }
