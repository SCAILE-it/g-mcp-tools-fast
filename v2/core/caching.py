"""Caching utilities for V2 API.

Provides in-memory caching with TTL support for scraper results.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


# In-memory cache storage
_cache: Dict[str, tuple[Any, datetime]] = {}
TTL_HOURS = 24


def _cache_key(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> str:
    """Generate a cache key from scraping parameters.

    Args:
        url: Target URL
        prompt: Extraction prompt
        schema: Optional output schema

    Returns:
        SHA256 hash of the combined parameters
    """
    schema_str = json.dumps(schema, sort_keys=True) if schema else ""
    combined = f"{url}|{prompt}|{schema_str}"
    return hashlib.sha256(combined.encode()).hexdigest()


def _get_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Retrieve cached result if available and not expired.

    Args:
        url: Target URL
        prompt: Extraction prompt
        schema: Optional output schema

    Returns:
        Cached value if found and valid, None otherwise
    """
    key = _cache_key(url, prompt, schema)
    if key in _cache:
        value, timestamp = _cache[key]
        if datetime.now() - timestamp < timedelta(hours=TTL_HOURS):
            return value
        # Expired - remove from cache
        del _cache[key]
    return None


def _set_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]], value: Any) -> None:
    """Store a value in the cache.

    Args:
        url: Target URL
        prompt: Extraction prompt
        schema: Optional output schema
        value: Value to cache
    """
    key = _cache_key(url, prompt, schema)
    _cache[key] = (value, datetime.now())


def clear_cache() -> None:
    """Clear all cached values."""
    global _cache
    _cache = {}


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dictionary with cache size and other stats
    """
    return {
        "size": len(_cache),
        "ttl_hours": TTL_HOURS
    }
