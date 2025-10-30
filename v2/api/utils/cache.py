"""Caching utilities for V2 API."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

_cache: Dict[str, tuple[Any, datetime]] = {}
TTL_HOURS = 24


def _cache_key(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> str:
    """Generate cache key from URL, prompt, and schema."""
    schema_str = json.dumps(schema, sort_keys=True) if schema else ""
    combined = f"{url}|{prompt}|{schema_str}"
    return hashlib.sha256(combined.encode()).hexdigest()


def get_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Get cached value if it exists and is not expired."""
    key = _cache_key(url, prompt, schema)
    if key in _cache:
        value, timestamp = _cache[key]
        if datetime.now() - timestamp < timedelta(hours=TTL_HOURS):
            return value
        del _cache[key]
    return None


def set_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]], value: Any) -> None:
    """Store value in cache with current timestamp."""
    key = _cache_key(url, prompt, schema)
    _cache[key] = (value, datetime.now())
