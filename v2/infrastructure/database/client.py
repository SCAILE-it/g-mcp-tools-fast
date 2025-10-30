"""Supabase client singleton for V2 API.

Fixes DRY violation: V1 had create_client() called 20+ times throughout codebase.
Singleton pattern ensures single client instance with proper connection pooling.
"""

import asyncio
from typing import Optional

from supabase import Client, create_client

from v2.config import config
from v2.core.logging import logger


class SupabaseClient:
    """Singleton Supabase client with lazy initialization."""

    _instance: Optional["SupabaseClient"] = None
    _lock = asyncio.Lock()
    _client: Optional[Client] = None

    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls) -> "SupabaseClient":
        """Get or create singleton instance (thread-safe)."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def client(self) -> Client:
        """Get Supabase client, initializing if needed."""
        if self._client is None:
            url = config.supabase_url()
            key = config.supabase_service_role_key()

            if not url or not key:
                raise RuntimeError(
                    "Supabase not configured. Set SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY environment variables."
                )

            self._client = create_client(url, key)
            logger.info("supabase_client_initialized", url=url)

        return self._client

    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return config.supabase_url() is not None and config.supabase_service_role_key() is not None
