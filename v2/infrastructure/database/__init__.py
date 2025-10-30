"""Database module for V2 API.

Provides Supabase client singleton and repository pattern for database operations.

Fixes V1 issues:
- DRY violation: create_client() called 20+ times → Singleton pattern
- Dependency Inversion: Direct table access → Repository pattern
- Type safety: Raw dicts → Typed dataclasses
"""

from v2.infrastructure.database.client import SupabaseClient
from v2.infrastructure.database.models import APICallRecord, SavedJob
from v2.infrastructure.database.repositories import (
    APICallRepository,
    BaseRepository,
    JobRepository,
    QuotaRepository,
)

__all__ = [
    "SupabaseClient",
    "APICallRecord",
    "SavedJob",
    "BaseRepository",
    "APICallRepository",
    "QuotaRepository",
    "JobRepository",
]
