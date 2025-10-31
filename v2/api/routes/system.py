"""System routes for V2 API."""

import asyncio
import os
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from v2.api.dependencies import get_tools_registry

router = APIRouter(tags=["System"])


async def test_gemini_connection() -> Dict[str, Any]:
    """Test Gemini API connectivity. Returns dict with status and details."""
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            return {"status": "unconfigured", "error": "GEMINI_API_KEY not set"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Minimal test request with 2-second timeout
        await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, "test"), timeout=2.0
        )

        return {"status": "healthy", "model": "gemini-2.0-flash-exp"}
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Request timed out after 2s"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)[:100]}


async def test_supabase_connection() -> Dict[str, Any]:
    """Test Supabase connectivity. Returns dict with status and details."""
    try:
        from supabase import create_client

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not url or not key:
            return {"status": "unconfigured", "error": "Supabase credentials not set"}

        # Test connection with simple query
        supabase = create_client(url, key)
        await asyncio.wait_for(
            asyncio.to_thread(lambda: supabase.table("api_calls").select("id").limit(1).execute()),
            timeout=2.0,
        )

        return {"status": "healthy", "database": "connected"}
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Request timed out after 2s"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)[:100]}


@router.get("/health")
async def health_check(tools_registry: Dict[str, Any] = Depends(get_tools_registry)):
    """Health check with Gemini/Supabase testing. Returns service status, version, tool count, and dependency health."""
    # Get tool categories from registry
    categories = list({config["type"] for config in tools_registry.values()}) if tools_registry else []
    tool_count = len(tools_registry)

    # Test dependencies in parallel
    results = await asyncio.gather(
        test_gemini_connection(), test_supabase_connection(), return_exceptions=True
    )

    gemini_health: Dict[str, Any]
    supabase_health: Dict[str, Any]

    # Handle exceptions from gather
    if isinstance(results[0], Exception):
        gemini_health = {"status": "error", "error": str(results[0])}
    else:
        gemini_health = results[0]  # type: ignore[assignment]

    if isinstance(results[1], Exception):
        supabase_health = {"status": "error", "error": str(results[1])}
    else:
        supabase_health = results[1]  # type: ignore[assignment]

    # Determine overall status
    all_healthy = (
        gemini_health.get("status") == "healthy" and supabase_health.get("status") == "healthy"
    )
    overall_status = "healthy" if all_healthy else "degraded"

    return {
        "status": overall_status,
        "service": "g-mcp-tools-fast",
        "version": "1.0.0",
        "tools": tool_count,
        "categories": categories,
        "dependencies": {"gemini": gemini_health, "supabase": supabase_health},
        "timestamp": datetime.now().isoformat() + "Z",
    }


@router.get("/debug/headers")
async def debug_headers(request: Request):
    """Inspect request headers for rate limiting. Returns headers and IP candidate sources."""
    headers = dict(request.headers)
    client = request.client

    return {
        "headers": headers,
        "client_host": client.host if client else None,
        "client_port": client.port if client else None,
        "real_ip_candidates": {
            "x-forwarded-for": headers.get("x-forwarded-for"),
            "x-real-ip": headers.get("x-real-ip"),
            "cf-connecting-ip": headers.get("cf-connecting-ip"),
            "true-client-ip": headers.get("true-client-ip"),
        },
    }
