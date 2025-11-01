"""Health check functions for V2 API.

Tests connectivity to external dependencies (Gemini, Supabase).
"""

import asyncio
from typing import Any, Dict

import google.generativeai as genai

from v2.config import config
from v2.infrastructure.database import SupabaseClient


async def test_gemini_connection() -> Dict[str, Any]:
    """Test Gemini API connectivity with minimal request.

    Returns:
        Dict with status ("healthy", "unconfigured", "timeout", "unavailable")
        and optional error message
    """
    try:
        api_key = config.gemini_api_key()

        if not api_key:
            return {"status": "unconfigured", "error": "GEMINI_API_KEY not set"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Minimal test request with 2-second timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, "test"), timeout=2.0
        )

        return {"status": "healthy", "model": "gemini-2.0-flash-exp"}

    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Request timed out after 2s"}

    except Exception as e:
        return {"status": "unavailable", "error": str(e)[:100]}


async def test_supabase_connection() -> Dict[str, Any]:
    """Test Supabase connectivity with minimal query.

    Returns:
        Dict with status ("healthy", "unconfigured", "timeout", "unavailable")
        and optional error message
    """
    try:
        if not config.supabase_url() or not config.supabase_service_role_key():
            return {"status": "unconfigured", "error": "Supabase credentials not set"}

        # Get client instance
        client = SupabaseClient()
        supabase = client.client

        # Test connection with simple query
        result = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: supabase.table("api_calls").select("id").limit(1).execute()
            ),
            timeout=2.0,
        )

        return {"status": "healthy", "database": "connected"}

    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Request timed out after 2s"}

    except Exception as e:
        return {"status": "unavailable", "error": str(e)[:100]}
