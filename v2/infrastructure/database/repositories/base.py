"""Base repository interface for V2 API.

Implements Repository pattern with Dependency Inversion principle.
All concrete repositories inherit from BaseRepository.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from v2.infrastructure.database.client import SupabaseClient

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository for database operations.

    Provides dependency inversion - depend on repository interface, not concrete tables.
    """

    def __init__(self):
        """Initialize repository with Supabase client."""
        self._client: SupabaseClient = SupabaseClient()

    @property
    def db(self):
        """Get Supabase client instance."""
        return self._client.client

    @abstractmethod
    def table_name(self) -> str:
        """Return the table name this repository manages."""
        pass
